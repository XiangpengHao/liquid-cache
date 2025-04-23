mod eviction_cache;
use clap::Parser;
use eviction_cache::{Cache, ClockCache, FifoCache, LfuCache, LruCache};
use itertools::Itertools;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the parquet trace file
    #[arg(short, long)]
    trace_path: PathBuf,
}

struct NumSum {
    name: String,
    vec: Vec<u64>,
}
impl NumSum {
    fn from_vec(name: String, vec: Vec<u64>) -> Self {
        Self { name, vec }
    }

    fn summ(&self) -> (u64, u64, f64, f64, u64) {
        let count = self.vec.len() as u64;
        let min = *self.vec.iter().min().unwrap_or(&0);
        let max = *self.vec.iter().max().unwrap_or(&0);
        let mean = self.vec.iter().sum::<u64>() as f64 / count as f64;
        let variance = self
            .vec
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let stdev = variance.sqrt();

        (min, max, mean, stdev, count)
    }
    fn summ_format(&self) -> String {
        let (min, max, mean, stdev, count) = self.summ();
        format!(
            "Summary Statistics ({}):\nmin: {}\nmax: {}\nmean: {}\nstdev: {}\ncount: {}\n",
            self.name, min, max, mean, stdev, count
        )
    }
}

fn pack_u16s(a: u16, b: u16, c: u16, d: u16) -> u64 {
    ((a as u64) << 48) | ((b as u64) << 32) | ((c as u64) << 16) | (d as u64)
}

fn bench<C: Cache>(create: impl Fn(u64) -> C, name: String, args: &Args) {
    let file = File::open(args.trace_path.clone()).expect("Failed to open parquet file");
    let parquet_reader = SerializedFileReader::new(file).expect("Failed to create parquet reader");
    let row_iter = parquet_reader
        .get_row_iter(None)
        .expect("Failed to get row iterator");

    // Read all rows into memory
    let rows: Vec<_> = row_iter
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to read rows");
    let total_rows = rows.len();
    println!("Total rows: {}", total_rows);

    let mut cache_size = total_rows;

    while cache_size > 0 {
        let mut cache = create(cache_size as u64);

        for row in &rows {
            // Retrieve the required fields by index:
            // 0: file_id, 1: row_group, 2: col, and 4: size.
            let file_id: u16 = row.get_ulong(0).expect("Failed to get file_id") as u16;
            let row_group_id: u16 = row.get_ulong(1).expect("Failed to get row_group_id") as u16;
            let column_id: u16 = row.get_ulong(2).expect("Failed to get column_id") as u16;
            let batch_id: u16 = row.get_ulong(3).expect("Failed to get batch_id") as u16;

            let cache_memory_bytes: u16 =
                row.get_ulong(4).expect("Failed to get cache_memory_bytes") as u16;
            //let time_stamp_nanos: u16 =
            //    row.get_ulong(5).expect("Failed to get time_stamp_nanos") as u16;

            let key = pack_u16s(file_id, row_group_id, column_id, batch_id);
            cache.get(key, cache_memory_bytes as u64);
        }

        let (hits, total) = cache.result();
        println!("{},{},{},{}", name, cache_size, hits, total);

        cache_size /= 10;
    }
}

fn access_patterns(args: &Args) {
    let file = File::open(args.trace_path.clone()).expect("Failed to open parquet file");
    let parquet_reader = SerializedFileReader::new(file).expect("Failed to create parquet reader");
    let row_iter = parquet_reader
        .get_row_iter(None)
        .expect("Failed to get row iterator");

    // Read all rows into memory
    let rows: Vec<_> = row_iter
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to read rows");

    // HashMap<key, (first_access_time, last_access_time, access_count)>
    let mut lifetime_map: HashMap<u64, (u64, u64, u32)> = HashMap::new();

    let mut prev: Option<(u16, u16, u16, u16)> = None;
    let mut dists = Vec::with_capacity(rows.len());

    for row in &rows {
        // Retrieve the required fields by index:
        // 0: file_id, 1: row_group, 2: col, and 4: size.
        let file_id: u16 = row.get_ulong(0).expect("Failed to get file_id") as u16;
        let row_group_id: u16 = row.get_ulong(1).expect("Failed to get row_group_id") as u16;
        let column_id: u16 = row.get_ulong(2).expect("Failed to get column_id") as u16;
        let batch_id: u16 = row.get_ulong(3).expect("Failed to get batch_id") as u16;

        //let cache_memory_bytes: u64 =
        //    row.get_ulong(4).expect("Failed to get cache_memory_bytes");
        let time_stamp_nanos: u64 = row.get_ulong(5).expect("Failed to get time_stamp_nanos");

        let key = pack_u16s(file_id, row_group_id, column_id, batch_id);

        if let Some(val) = lifetime_map.get_mut(&key) {
            val.1 = time_stamp_nanos;
            val.2 += 1;
        } else {
            lifetime_map.insert(key, (time_stamp_nanos, time_stamp_nanos, 1));
        }

        let tup = (file_id, row_group_id, column_id, batch_id);
        if let Some((pf, pg, pc, pb)) = prev {
            // Manhattan distance in 4‑D space:
            let dist = (file_id as i64 != pf as i64) as u64
                + (row_group_id as i64 != pg as i64) as u64
                + (column_id as i64 != pc as i64) as u64
                + (batch_id as i64 != pb as i64) as u64;
            dists.push(dist);
        }
        prev = Some(tup);
    }

    //let access_summary = NumSum::from_vec("Access Distance".to_string(), dists);
    let lifetime_summary = NumSum::from_vec(
        "Lifetime [ms]".to_string(),
        lifetime_map
            .values()
            .map(|(a, b, _)| (*b - *a) / 1_000_000)
            .collect(),
    );

    let res: Vec<u32> = lifetime_map
        .values()
        .map(|(_, _, count)| *count)
        .sorted()
        .collect();

    let mut writer =
        csv::Writer::from_path("access_counts.csv").expect("Failed to create CSV file");
    writer
        .write_record(["Access Count"])
        .expect("Failed to write header");
    for count in res {
        writer
            .write_record(&[count.to_string()])
            .expect("Failed to write record");
    }
    writer.flush().expect("Failed to flush CSV file");

    println!("Total rows: {}", rows.len());
    println!("Unique rows: {}", lifetime_map.len());
    //println!("{}", access_summary.summ_format());
    println!("{}", lifetime_summary.summ_format());
}

fn main() {
    let args = Args::parse();
    // Run bench tests for each cache type.

    bench(LruCache::new, "LRU".to_string(), &args);
    bench(ClockCache::new, "CLOCK".to_string(), &args);
    bench(LfuCache::new, "LFU".to_string(), &args);
    bench(FifoCache::new, "FIFO".to_string(), &args);

    access_patterns(&args);
}
