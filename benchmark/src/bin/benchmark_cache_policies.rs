use clap::Parser;
use liquid_cache_storage::cache::{CachedBatchType, EntryID};
use liquid_cache_storage::cache_policies::{
    ClockPolicy, FifoPolicy, FiloPolicy, LiquidPolicy, LruPolicy, S3FifoPolicy, SievePolicy,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
struct Args {
    /// Directory containing parquet trace files (entry_id, entry_size, cache_memory_bytes, time_stamp_nanos)
    #[arg(long, default_value = "trace")]
    trace_dir: PathBuf,

    /// Output CSV path for results
    #[arg(long, default_value = "benchmark/data/cache_policy_results.csv")]
    output: PathBuf,

    /// Number of cache sizes to sweep linearly from small to large
    #[arg(long, default_value_t = 6)]
    sizes: usize,

    /// Optional path to write a Mermaid xychart snippet (hit ratio vs policy)
    #[arg(long)]
    mermaid: Option<PathBuf>,
}

fn list_parquet_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("cannot read dir {}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let is_parquet = p.extension().map(|e| e == "parquet").unwrap_or(false);
            let name_ok = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("cache-trace-"))
                .unwrap_or(false);
            is_parquet && name_ok
        })
        .collect();
    files.sort();
    files
}

#[derive(Clone, Copy, Debug)]
struct TraceEvent {
    id: u64,
    size: u64,
}

fn read_trace(dir: &Path) -> Vec<TraceEvent> {
    let mut out: Vec<TraceEvent> = Vec::new();
    for path in list_parquet_files(dir) {
        let file = std::fs::File::open(&path).expect("open parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("reader")
            .build()
            .expect("build");
        for batch in reader {
            let batch = batch.expect("batch");
            let entry_id = batch.column_by_name("entry_id").unwrap_or_else(|| {
                let cols: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                panic!(
                    "missing entry_id column in {:?}; available columns: {:?}",
                    path, cols
                )
            });
            let entry_size = batch.column_by_name("entry_size").unwrap_or_else(|| {
                let cols: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                panic!(
                    "missing entry_size column in {:?}; available columns: {:?}",
                    path, cols
                )
            });
            let ids = arrow::array::as_primitive_array::<arrow::datatypes::UInt64Type>(entry_id);
            let sizes =
                arrow::array::as_primitive_array::<arrow::datatypes::UInt64Type>(entry_size);
            for i in 0..batch.num_rows() {
                out.push(TraceEvent {
                    id: ids.value(i),
                    size: sizes.value(i),
                });
            }
        }
    }
    out
}

fn compute_total_unique_size(events: &[TraceEvent]) -> u64 {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut total: u64 = 0;
    for ev in events.iter() {
        if seen.insert(ev.id) {
            total = total.saturating_add(ev.size);
        }
    }
    total
}

trait PolicyLike: std::fmt::Debug + Send + Sync {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID>;
    fn notify_insert(&self, entry_id: &EntryID, batch: CachedBatchType);
    fn notify_access(&self, entry_id: &EntryID, batch: CachedBatchType);
}

impl<T: liquid_cache_storage::cache_policies::CachePolicy + std::fmt::Debug + Send + Sync>
    PolicyLike for T
{
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        liquid_cache_storage::cache_policies::CachePolicy::find_victim(self, cnt)
    }
    fn notify_insert(&self, entry_id: &EntryID, batch: CachedBatchType) {
        liquid_cache_storage::cache_policies::CachePolicy::notify_insert(self, entry_id, batch)
    }
    fn notify_access(&self, entry_id: &EntryID, batch: CachedBatchType) {
        liquid_cache_storage::cache_policies::CachePolicy::notify_access(self, entry_id, batch)
    }
}

fn simulate_policy(
    policy: &dyn PolicyLike,
    events: &[TraceEvent],
    budget_bytes: u64,
) -> (u64, u64) {
    let mut used_bytes: u64 = 0;
    let mut index: HashMap<u64, u64> = HashMap::new();
    let mut hits: u64 = 0;
    let mut total: u64 = 0;
    let batch_ty = CachedBatchType::MemoryArrow;

    for ev in events.iter() {
        total += 1;
        let entry = EntryID::from(ev.id as usize);
        if index.contains_key(&ev.id) {
            hits += 1;
            policy.notify_access(&entry, batch_ty);
            continue;
        }

        while used_bytes + ev.size > budget_bytes && !index.is_empty() {
            let victims = policy.find_victim(1);
            if victims.is_empty() {
                break;
            }
            for v in victims.into_iter() {
                let victim_id = usize::from(v) as u64;
                if let Some(sz) = index.remove(&victim_id) {
                    used_bytes = used_bytes.saturating_sub(sz);
                }
            }
        }

        if ev.size <= budget_bytes {
            index.insert(ev.id, ev.size);
            used_bytes = used_bytes.saturating_add(ev.size);
            policy.notify_insert(&entry, batch_ty);
        }
    }

    (hits, total)
}

fn build_policies() -> Vec<(&'static str, Box<dyn PolicyLike>)> {
    vec![
        ("LRU", Box::new(LruPolicy::new())),
        ("CLOCK", Box::new(ClockPolicy::new())),
        ("FIFO", Box::new(FifoPolicy::new())),
        ("FILO", Box::new(FiloPolicy::new())),
        ("S3FIFO", Box::new(S3FifoPolicy::new())),
        ("SIEVE", Box::new(SievePolicy::new())),
        ("LIQUID3Q", Box::new(LiquidPolicy::default())),
    ]
}

fn sweep_budgets(total_unique_size: u64, steps: usize) -> Vec<u64> {
    if steps == 0 || total_unique_size == 0 {
        return vec![];
    }
    let step = std::cmp::max(1u64, total_unique_size / steps as u64);
    let mut out = Vec::with_capacity(steps);
    let mut b = step;
    for _ in 0..steps {
        out.push(b);
        b = b.saturating_add(step);
        if b > total_unique_size {
            b = total_unique_size;
        }
    }
    out
}

fn build_mermaid_chart(points: &[(String, u64, f64)], total_unique_size: u64) -> Option<String> {
    if points.is_empty() || total_unique_size == 0 {
        return None;
    }

    let mut budgets: Vec<u64> = points
        .iter()
        .map(|(_, cache_size, _)| *cache_size)
        .collect();
    budgets.sort_unstable();
    budgets.dedup();
    if budgets.is_empty() {
        return None;
    }

    let mut policies: Vec<String> = Vec::new();
    let mut policy_index: HashMap<String, usize> = HashMap::new();
    let mut policy_values: Vec<HashMap<u64, f64>> = Vec::new();

    for (policy, cache_size, ratio) in points.iter() {
        let idx = *policy_index.entry(policy.clone()).or_insert_with(|| {
            let idx = policies.len();
            policies.push(policy.clone());
            policy_values.push(HashMap::new());
            idx
        });
        policy_values[idx].insert(*cache_size, *ratio);
    }

    fn fmt_float(value: f64) -> String {
        let mut s = format!("{value:.2}");
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    }

    let axis_labels: Vec<String> = budgets
        .iter()
        .map(|cache_size| {
            let pct = (*cache_size as f64 / total_unique_size as f64) * 100.0;
            format!("\"{}\"", fmt_float(pct))
        })
        .collect();

    let mut out = String::new();
    out.push_str("%% Cache policy hit ratio chart generated by benchmark_cache_policies\n");
    out.push_str("xychart-beta\n");
    out.push_str("    title \"Hit Ratio vs Cache Size\"\n");
    out.push_str(&format!(
        "    x-axis \"Cache Size (% of working set)\" [{}]\n",
        axis_labels.join(", ")
    ));
    out.push_str("    y-axis \"Hit Ratio (%)\" 0 --> 100\n");

    for (policy_idx, policy_name) in policies.iter().enumerate() {
        let values_map = &policy_values[policy_idx];
        let series_values: Vec<String> = budgets
            .iter()
            .map(|cache_size| {
                let pct = values_map
                    .get(cache_size)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    .min(1.0)
                    * 100.0;
                fmt_float(pct)
            })
            .collect();
        out.push_str(&format!(
            "    line \"{}\" [{}]\n",
            policy_name,
            series_values.join(", ")
        ));
    }

    Some(out)
}

fn main() {
    let args = Args::parse();
    let dir = &args.trace_dir;
    eprintln!("Reading traces from {}", dir.display());
    let events = read_trace(dir);
    let total_unique_size = compute_total_unique_size(&events);
    eprintln!("Total unique size: {} bytes", total_unique_size);

    let budgets = sweep_budgets(total_unique_size, args.sizes);
    if budgets.is_empty() {
        eprintln!("No budgets to test");
        return;
    }

    let mut wtr = csv::Writer::from_path(&args.output).expect("open output");
    wtr.write_record(["policy", "cache_bytes", "hits", "total", "hit_ratio"])
        .unwrap();

    let mut mermaid_points: Vec<(String, u64, f64)> = Vec::new();
    let policies = build_policies();
    for budget in budgets.iter() {
        for (label, policy) in policies.iter() {
            let (hits, total) = simulate_policy(policy.as_ref(), &events, *budget);
            let ratio = if total == 0 {
                0.0
            } else {
                hits as f64 / total as f64
            };
            let record = vec![
                label.to_string(),
                budget.to_string(),
                hits.to_string(),
                total.to_string(),
                ratio.to_string(),
            ];
            wtr.write_record(record).unwrap();
            // Collect points for all budgets - (policy, cache_size, hit_ratio)
            mermaid_points.push((label.to_string(), *budget, ratio));
        }
        wtr.flush().unwrap();
    }

    if let Some(mermaid_path) = &args.mermaid {
        if let Some(chart) = build_mermaid_chart(&mermaid_points, total_unique_size) {
            match std::fs::write(mermaid_path, chart) {
                Ok(_) => eprintln!("Mermaid chart written to {}", mermaid_path.display()),
                Err(e) => eprintln!("Failed to write Mermaid chart {:?}: {}", mermaid_path, e),
            }
        } else {
            eprintln!("Not enough data to build Mermaid chart");
        }
    }

    eprintln!("Results written to {}", args.output.display());
}
