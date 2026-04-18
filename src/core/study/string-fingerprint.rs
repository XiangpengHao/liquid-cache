use std::fs;
use std::path::PathBuf;

use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueEnum};
use datafusion::prelude::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fingerprint {
    words: [u64; 4],
}

impl Fingerprint {
    #[inline]
    fn from_bytes(bytes: &[u8], mapping: &ByteBucketMap) -> Self {
        let mut words = [0u64; 4];
        for &b in bytes {
            let bucket = mapping.table[b as usize] as usize;
            debug_assert!(bucket < mapping.bucket_count as usize);
            words[bucket >> 6] |= 1u64 << (bucket & 63);
        }
        Self { words }
    }

    #[inline]
    fn might_contain(&self, pattern: &Self) -> bool {
        (self.words[0] & pattern.words[0]) == pattern.words[0]
            && (self.words[1] & pattern.words[1]) == pattern.words[1]
            && (self.words[2] & pattern.words[2]) == pattern.words[2]
            && (self.words[3] & pattern.words[3]) == pattern.words[3]
    }
}

#[derive(Debug, Clone)]
struct ByteBucketMap {
    bucket_count: u16,
    table: [u8; 256],
}

impl ByteBucketMap {
    fn round_robin(bucket_count: u16) -> Self {
        assert!((1..=256).contains(&bucket_count));
        let mut table = [0u8; 256];
        for (b, slot) in table.iter_mut().enumerate() {
            *slot = (b as u16 % bucket_count) as u8;
        }
        Self {
            bucket_count,
            table,
        }
    }

    fn contiguous_range(bucket_count: u16) -> Self {
        assert!((1..=256).contains(&bucket_count));
        let mut table = [0u8; 256];
        for (b, slot) in table.iter_mut().enumerate() {
            *slot = (((b as u16) * bucket_count) >> 8) as u8;
        }
        Self {
            bucket_count,
            table,
        }
    }

    #[allow(dead_code)]
    fn from_fn(bucket_count: u16, assign: impl Fn(u8) -> u8) -> Self {
        assert!((1..=256).contains(&bucket_count));
        let mut table = [0u8; 256];
        for (b, slot) in table.iter_mut().enumerate() {
            let bucket = assign(b as u8);
            assert!(
                (bucket as u16) < bucket_count,
                "custom bucket assignment returned {bucket} (bucket_count={bucket_count})"
            );
            *slot = bucket;
        }
        Self {
            bucket_count,
            table,
        }
    }

    fn from_table(bucket_count: u16, table: [u8; 256]) -> Self {
        assert!((1..=256).contains(&bucket_count));
        for (b, &bucket) in table.iter().enumerate() {
            assert!(
                (bucket as u16) < bucket_count,
                "custom mapping table[{b}]={bucket} out of range (bucket_count={bucket_count})"
            );
        }
        Self {
            bucket_count,
            table,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MappingKind {
    RoundRobin,
    ContiguousRange,
    CustomTable,
    Optimized,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "String Fingerprint Study")]
#[command(about = "Compute string fingerprints and evaluate pattern filtering effectiveness")]
struct CliArgs {
    /// Parquet file to read.
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Columns to process.
    #[arg(long, value_delimiter = ',', default_value = "URL,Title,Referer")]
    columns: Vec<String>,

    /// Pattern to test (byte-based, default: google).
    #[arg(long, default_value = "google")]
    pattern: String,

    /// Number of buckets (n). More buckets => longer fingerprint (up to 256).
    #[arg(long, value_delimiter = ',', default_value = "4,8,16,32,64")]
    buckets: Vec<u16>,

    /// Sweep bucket counts from start to end (inclusive).
    #[arg(long)]
    bucket_start: Option<u16>,

    /// Sweep bucket counts from start to end (inclusive).
    #[arg(long)]
    bucket_end: Option<u16>,

    /// Step for --bucket-start/--bucket-end.
    #[arg(long, default_value_t = 1)]
    bucket_step: u16,

    /// Bucket mapping strategy.
    #[arg(long, value_enum, default_value_t = MappingKind::RoundRobin)]
    mapping: MappingKind,

    /// Path to a custom mapping table (256 integers in [0, n), separated by whitespace or commas).
    #[arg(long)]
    custom_map: Option<PathBuf>,

    /// For `--mapping optimized`: number of (non-null) values sampled from the start of the column.
    #[arg(long, default_value_t = 100)]
    sample_values: usize,

    /// Optional row limit per column (useful for faster runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors.
    #[arg(long, default_value = "false")]
    bench: bool,
}

#[derive(Default, Debug, Clone)]
struct Stats {
    rows: usize,
    nulls: usize,
    filtered_out: usize,
    candidates: usize,
    false_pos: usize,
    actual_present: usize,
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.rows += other.rows;
        self.nulls += other.nulls;
        self.filtered_out += other.filtered_out;
        self.candidates += other.candidates;
        self.false_pos += other.false_pos;
        self.actual_present += other.actual_present;
    }
}

#[tokio::main]
async fn main() {
    let args = CliArgs::parse();

    let mut config = SessionConfig::default().with_batch_size(8192 * 2);
    let options = config.options_mut();
    options.execution.parquet.schema_force_view_types = false;

    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("hits", &args.parquet, Default::default())
        .await
        .expect("register parquet");

    println!(
        "| Column | Pattern | Gram | Mapping | n | Rows | Nulls | Filtered Out | % | Candidates | % | False Pos | % | Actual Present | % |"
    );
    println!(
        "|--------|---------|------|---------|---|------|-------|--------------|---|------------|---|-----------|---|----------------|---|"
    );

    let bucket_counts = resolve_bucket_counts(&args);

    for column in &args.columns {
        let sql = if let Some(limit) = args.limit {
            format!("SELECT \"{column}\" FROM \"hits\" LIMIT {limit}")
        } else {
            format!("SELECT \"{column}\" FROM \"hits\"")
        };
        let df = ctx.sql(&sql).await.expect("create df");
        let batches = df.collect().await.expect("collect");

        let (histogram, pattern_unique) = if matches!(args.mapping, MappingKind::Optimized) {
            (
                sample_histogram_from_batches(&batches, args.sample_values),
                unique_bytes(args.pattern.as_bytes()),
            )
        } else {
            ([0u64; 256], [false; 256])
        };

        for &bucket_count in &bucket_counts {
            let mapping = build_mapping(
                bucket_count,
                args.mapping,
                args.custom_map.as_ref(),
                &histogram,
                &pattern_unique,
            );
            let pattern_fp = Fingerprint::from_bytes(args.pattern.as_bytes(), &mapping);

            let mut stats = Stats::default();
            for batch in &batches {
                let array: ArrayRef = batch.column(0).clone();
                stats.add(&eval_array(&array, &args.pattern, &mapping, &pattern_fp));
            }

            let non_null = stats.rows.saturating_sub(stats.nulls);
            let filtered_pct = pct(stats.filtered_out, non_null);
            let candidates_pct = pct(stats.candidates, non_null);
            let false_pos_pct = pct(stats.false_pos, stats.candidates);
            let actual_present_pct = pct(stats.actual_present, non_null);

            println!(
                "| {column} | {} | One | {} | {bucket_count} | {} | {} | {} | {:.2}% | {} | {:.2}% | {} | {:.2}% | {} | {:.2}% |",
                args.pattern,
                mapping_label(args.mapping),
                stats.rows,
                stats.nulls,
                stats.filtered_out,
                filtered_pct,
                stats.candidates,
                candidates_pct,
                stats.false_pos,
                false_pos_pct,
                stats.actual_present,
                actual_present_pct,
            );
        }
    }
}

fn resolve_bucket_counts(args: &CliArgs) -> Vec<u16> {
    match (args.bucket_start, args.bucket_end) {
        (Some(start), Some(end)) => {
            assert!(args.bucket_step > 0, "bucket_step must be > 0");
            assert!(
                (1..=256).contains(&start) && (1..=256).contains(&end),
                "bucket_start and bucket_end must be in 1..=256"
            );
            assert!(start <= end, "bucket_start must be <= bucket_end");
            let mut buckets = Vec::new();
            let mut value = start;
            while value <= end {
                buckets.push(value);
                let next = value.saturating_add(args.bucket_step);
                if next == value {
                    break;
                }
                value = next;
            }
            buckets
        }
        (None, None) => args.buckets.clone(),
        _ => panic!("bucket_start and bucket_end must be set together"),
    }
}

fn mapping_label(kind: MappingKind) -> &'static str {
    match kind {
        MappingKind::RoundRobin => "RoundRobin",
        MappingKind::ContiguousRange => "ContiguousRange",
        MappingKind::CustomTable => "CustomTable",
        MappingKind::Optimized => "Optimized",
    }
}

fn pct(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        (numer as f64) * 100.0 / (denom as f64)
    }
}

fn build_mapping(
    bucket_count: u16,
    kind: MappingKind,
    custom_map: Option<&PathBuf>,
    histogram: &[u64; 256],
    pattern_unique: &[bool; 256],
) -> ByteBucketMap {
    match kind {
        MappingKind::RoundRobin => ByteBucketMap::round_robin(bucket_count),
        MappingKind::ContiguousRange => ByteBucketMap::contiguous_range(bucket_count),
        MappingKind::CustomTable => {
            let path = custom_map.expect("CustomTable mapping requires --custom-map <path>");
            let table = read_custom_table(path);
            ByteBucketMap::from_table(bucket_count, table)
        }
        MappingKind::Optimized => {
            let table = optimized_mapping_table(bucket_count, histogram, pattern_unique);
            ByteBucketMap::from_table(bucket_count, table)
        }
    }
}

fn unique_bytes(bytes: &[u8]) -> [bool; 256] {
    let mut present = [false; 256];
    for &b in bytes {
        present[b as usize] = true;
    }
    present
}

fn sample_histogram_from_batches(batches: &[RecordBatch], sample_values: usize) -> [u64; 256] {
    let mut histogram = [0u64; 256];
    let mut remaining = sample_values;
    if remaining == 0 {
        return histogram;
    }

    for batch in batches {
        if remaining == 0 {
            break;
        }
        let array = batch.column(0);
        add_histogram_from_array(array, &mut histogram, &mut remaining);
    }
    histogram
}

fn add_histogram_from_array(
    array: &ArrayRef,
    histogram: &mut [u64; 256],
    remaining_values: &mut usize,
) {
    if *remaining_values == 0 {
        return;
    }
    match array.data_type() {
        DataType::Utf8 => add_histogram_from_string_iter(
            array.as_string::<i32>().iter(),
            histogram,
            remaining_values,
        ),
        DataType::LargeUtf8 => add_histogram_from_string_iter(
            array.as_string::<i64>().iter(),
            histogram,
            remaining_values,
        ),
        DataType::Utf8View => add_histogram_from_string_iter(
            array.as_string_view().iter(),
            histogram,
            remaining_values,
        ),
        DataType::Binary => add_histogram_from_binary_iter(
            array.as_binary::<i32>().iter(),
            histogram,
            remaining_values,
        ),
        DataType::LargeBinary => add_histogram_from_binary_iter(
            array.as_binary::<i64>().iter(),
            histogram,
            remaining_values,
        ),
        DataType::BinaryView => add_histogram_from_binary_iter(
            array.as_binary_view().iter(),
            histogram,
            remaining_values,
        ),
        other => panic!("unsupported data type for histogram sampling: {other:?}"),
    }
}

fn add_histogram_from_string_iter<'a>(
    iter: impl Iterator<Item = Option<&'a str>>,
    histogram: &mut [u64; 256],
    remaining_values: &mut usize,
) {
    for value in iter {
        if *remaining_values == 0 {
            break;
        }
        let Some(s) = value else {
            continue;
        };
        for &b in s.as_bytes() {
            histogram[b as usize] += 1;
        }
        *remaining_values -= 1;
    }
}

fn add_histogram_from_binary_iter<'a>(
    iter: impl Iterator<Item = Option<&'a [u8]>>,
    histogram: &mut [u64; 256],
    remaining_values: &mut usize,
) {
    for value in iter {
        if *remaining_values == 0 {
            break;
        }
        let Some(bytes) = value else {
            continue;
        };
        for &b in bytes {
            histogram[b as usize] += 1;
        }
        *remaining_values -= 1;
    }
}

fn optimized_mapping_table(
    bucket_count: u16,
    histogram: &[u64; 256],
    pattern_unique: &[bool; 256],
) -> [u8; 256] {
    assert!((1..=256).contains(&bucket_count));
    let bucket_count_usize = bucket_count as usize;
    assert!(bucket_count_usize > 0);

    let mut table = [0u8; 256];
    let mut assigned = [false; 256];
    let mut bucket_load = vec![0u64; bucket_count_usize];
    let mut pattern_bucket_used = vec![false; bucket_count_usize];
    let mut cursor = 0usize;

    // Assign distinct pattern bytes to distinct, currently-lowest-mass buckets.
    for b in 0u16..=255 {
        let b = b as usize;
        if !pattern_unique[b] {
            continue;
        }
        let bucket = if pattern_bucket_used.iter().all(|&v| v) {
            // More distinct pattern bytes than buckets: fall back to standard min-load assignment.
            min_load_bucket_from(&bucket_load, cursor)
        } else {
            min_load_bucket_excluding(&bucket_load, &pattern_bucket_used, cursor)
        };
        table[b] = bucket as u8;
        assigned[b] = true;
        pattern_bucket_used[bucket] = true;
        bucket_load[bucket] += histogram[b];
        cursor = (bucket + 1) % bucket_count_usize;
    }

    // Sort remaining bytes by descending frequency and greedily balance bucket loads.
    let mut remaining: Vec<u8> = (0u16..=255)
        .map(|b| b as u8)
        .filter(|&b| !assigned[b as usize])
        .collect();
    remaining.sort_unstable_by(|&a, &b| {
        let fa = histogram[a as usize];
        let fb = histogram[b as usize];
        fb.cmp(&fa).then_with(|| a.cmp(&b))
    });

    for b in remaining {
        let bucket = min_load_bucket_from(&bucket_load, cursor);
        cursor = (bucket + 1) % bucket_count_usize;
        let idx = b as usize;
        table[idx] = bucket as u8;
        bucket_load[bucket] += histogram[idx];
    }

    table
}

#[inline]
fn min_load_bucket_from(bucket_load: &[u64], start: usize) -> usize {
    let n = bucket_load.len();
    debug_assert!(n > 0);
    let start = start % n;
    let mut min_idx = start;
    let mut min_val = bucket_load[start];
    for offset in 1..n {
        let i = (start + offset) % n;
        let v = bucket_load[i];
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }
    min_idx
}

#[inline]
fn min_load_bucket_excluding(bucket_load: &[u64], excluded: &[bool], start: usize) -> usize {
    debug_assert_eq!(bucket_load.len(), excluded.len());
    let n = bucket_load.len();
    debug_assert!(n > 0);
    let start = start % n;

    let mut min_idx = None;
    let mut min_val = 0u64;
    for offset in 0..n {
        let i = (start + offset) % n;
        if excluded[i] {
            continue;
        }
        let v = bucket_load[i];
        match min_idx {
            None => {
                min_idx = Some(i);
                min_val = v;
            }
            Some(_) if v < min_val => {
                min_idx = Some(i);
                min_val = v;
            }
            _ => {}
        }
    }
    min_idx.expect("excluded all buckets")
}

fn read_custom_table(path: &PathBuf) -> [u8; 256] {
    let raw = fs::read_to_string(path).expect("read custom map");
    let tokens = raw
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty());
    let mut table = [0u8; 256];
    let mut i = 0usize;
    for tok in tokens {
        assert!(i < 256, "custom map has more than 256 values: {path:?}");
        let v: u16 = tok
            .parse()
            .unwrap_or_else(|_| panic!("invalid integer token '{tok}' in {path:?}"));
        assert!(
            v <= 255,
            "custom map value {v} out of range (0..=255): {path:?}"
        );
        table[i] = v as u8;
        i += 1;
    }
    assert!(
        i == 256,
        "custom map must contain exactly 256 values, got {i}: {path:?}"
    );
    table
}

fn eval_array(
    array: &ArrayRef,
    pattern: &str,
    mapping: &ByteBucketMap,
    pattern_fp: &Fingerprint,
) -> Stats {
    match array.data_type() {
        DataType::Utf8 => eval_string_iter(
            array.as_string::<i32>().iter(),
            pattern,
            mapping,
            pattern_fp,
        ),
        DataType::LargeUtf8 => eval_string_iter(
            array.as_string::<i64>().iter(),
            pattern,
            mapping,
            pattern_fp,
        ),
        DataType::Utf8View => {
            eval_string_iter(array.as_string_view().iter(), pattern, mapping, pattern_fp)
        }
        DataType::Binary => eval_binary_iter(
            array.as_binary::<i32>().iter(),
            pattern,
            mapping,
            pattern_fp,
        ),
        DataType::LargeBinary => eval_binary_iter(
            array.as_binary::<i64>().iter(),
            pattern,
            mapping,
            pattern_fp,
        ),
        DataType::BinaryView => {
            eval_binary_iter(array.as_binary_view().iter(), pattern, mapping, pattern_fp)
        }
        other => panic!("unsupported data type for string fingerprint study: {other:?}"),
    }
}

fn eval_string_iter<'a>(
    iter: impl Iterator<Item = Option<&'a str>>,
    pattern: &str,
    mapping: &ByteBucketMap,
    pattern_fp: &Fingerprint,
) -> Stats {
    let mut stats = Stats::default();
    for value in iter {
        stats.rows += 1;
        let Some(s) = value else {
            stats.nulls += 1;
            continue;
        };
        let fp = Fingerprint::from_bytes(s.as_bytes(), mapping);
        if !fp.might_contain(pattern_fp) {
            stats.filtered_out += 1;
            continue;
        }
        stats.candidates += 1;
        if s.contains(pattern) {
            stats.actual_present += 1;
        } else {
            stats.false_pos += 1;
        }
    }
    stats
}

fn eval_binary_iter<'a>(
    iter: impl Iterator<Item = Option<&'a [u8]>>,
    pattern: &str,
    mapping: &ByteBucketMap,
    pattern_fp: &Fingerprint,
) -> Stats {
    let mut stats = Stats::default();
    for value in iter {
        stats.rows += 1;
        let Some(bytes) = value else {
            stats.nulls += 1;
            continue;
        };
        let fp = Fingerprint::from_bytes(bytes, mapping);
        if !fp.might_contain(pattern_fp) {
            stats.filtered_out += 1;
            continue;
        }
        stats.candidates += 1;
        let present = std::str::from_utf8(bytes)
            .map(|s| s.contains(pattern))
            .unwrap_or(false);
        if present {
            stats.actual_present += 1;
        } else {
            stats.false_pos += 1;
        }
    }
    stats
}
