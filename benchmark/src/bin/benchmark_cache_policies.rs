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

    /// Optional path to write an SVG plot (hit ratio vs policy)
    #[arg(long)]
    plot: Option<PathBuf>,
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

fn draw_plot(
    output: &Path,
    points: &[(String, u64, f64)],
    total_unique_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::backend::BitMapBackend;
    use plotters::prelude::*;

    if points.is_empty() || total_unique_size == 0 {
        return Ok(());
    }

    // Convert once to percent and group by policy while preserving first-seen order
    let mut policies: Vec<String> = Vec::new();
    let mut series_per_policy: Vec<Vec<(f64, f64)>> = Vec::new();
    for (policy, cache_size, ratio) in points.iter() {
        let cache_pct = (*cache_size as f64 / total_unique_size as f64) * 100.0;
        let hit_pct = *ratio * 100.0;
        if let Some(pos) = policies.iter().position(|p| p == policy) {
            series_per_policy[pos].push((cache_pct, hit_pct));
        } else {
            policies.push(policy.clone());
            series_per_policy.push(vec![(cache_pct, hit_pct)]);
        }
    }

    let height = 400u32;
    let width = 800u32;

    let root = BitMapBackend::new(output, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Hit Ratio vs Cache Size", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .set_label_area_size(LabelAreaPosition::Right, 120)
        .build_cartesian_2d(0.0f64..100.0f64, 0.0f64..100.0f64)?;

    chart
        .configure_mesh()
        .y_desc("Hit Ratio (%)")
        .x_desc("Cache Size (% of Working Set)")
        .y_labels(10)
        .x_labels(10)
        .x_label_formatter(&|x| format!("{:.0}%", x))
        .y_label_formatter(&|y| format!("{:.0}%", y))
        .draw()?;

    // Draw points per policy (no connecting lines)
    for (policy_idx, policy_points) in series_per_policy.iter_mut().enumerate() {
        // Ensure monotonic x for nicer visuals
        policy_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let points_style = Palette99::pick(policy_idx).filled();
        let legend_style = Palette99::pick(policy_idx).stroke_width(2);

        chart
            .draw_series(
                policy_points
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 4, points_style)),
            )?
            .label(policies[policy_idx].clone())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], legend_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
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

    let mut plot_points: Vec<(String, u64, f64)> = Vec::new();
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
            // Collect points for ALL budgets - (policy, cache_size, hit_ratio)
            plot_points.push((label.to_string(), *budget, ratio));
        }
        wtr.flush().unwrap();
    }

    if let Some(plot_path) = &args.plot {
        if let Err(e) = draw_plot(plot_path, &plot_points, total_unique_size) {
            eprintln!("Failed to write plot {:?}: {}", plot_path, e);
        } else {
            eprintln!("Plot written to {}", plot_path.display());
        }
    }

    eprintln!("Results written to {}", args.output.display());
}
