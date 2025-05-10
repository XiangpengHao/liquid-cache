/*mod eviction_cache;
use clap::Parser;
use eviction_cache::{Cache, ClockCache, FifoCache, LfuCache, LruCache};
use itertools::Itertools;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use liquid_cache_parquet::cache::tracer::CacheAccessReason;


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

        let reason: u8 = row.get_ubyte(6).expect("Failed to get reason");
        let reason: CacheAccessReason = reason.into();
        println!("reason: {:?}", reason);

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

    /*bench(LruCache::new, "LRU".to_string(), &args);
    bench(ClockCache::new, "CLOCK".to_string(), &args);
    bench(LfuCache::new, "LFU".to_string(), &args);
    bench(FifoCache::new, "FIFO".to_string(), &args);*/

    access_patterns(&args);
}
*/

use clap::Parser;
use liquid_cache_parquet::cache::tracer::CacheAccessReason;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use plotters::prelude::*;
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

/// Pack four u16 identifiers into a single u64 key
fn pack_u16s(a: u16, b: u16, c: u16, d: u16) -> u64 {
    ((a as u64) << 48) | ((b as u64) << 32) | ((c as u64) << 16) | (d as u64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Open Parquet trace
    let file = File::open(&args.trace_path)?;
    let reader = SerializedFileReader::new(file)?;
    let rows = reader.get_row_iter(None)?.collect::<Result<Vec<_>, _>>()?;

    // Map: entry_key -> (reason -> count)
    let mut reason_map: HashMap<u64, HashMap<CacheAccessReason, u32>> = HashMap::new();

    for row in &rows {
        let file_id: u16 = row.get_ulong(0)? as u16;
        let row_group_id: u16 = row.get_ulong(1)? as u16;
        let column_id: u16 = row.get_ulong(2)? as u16;
        let batch_id: u16 = row.get_ulong(3)? as u16;
        let reason_byte: u8 = row.get_ubyte(6)?;
        let reason: CacheAccessReason = reason_byte.into();

        let key = pack_u16s(file_id, row_group_id, column_id, batch_id);

        let entry = reason_map.entry(key).or_default();
        *entry.entry(reason).or_insert(0) += 1;
    }

    // Build counts_by_reason: reason -> Vec<count_per_entry>
    let mut counts_by_reason: HashMap<CacheAccessReason, Vec<u32>> = HashMap::new();
    for entry_counts in reason_map.values() {
        for (&reason, &count) in entry_counts {
            counts_by_reason.entry(reason).or_default().push(count);
        }
    }

    plot_access_histogram(&counts_by_reason, "access_histogram.png")?;
    println!("Wrote histogram to access_histogram.png");
    Ok(())
}

/// Draws a stacked histogram of per-entry access counts by reason.
fn plot_access_histogram(
    counts_by_reason: &HashMap<CacheAccessReason, Vec<u32>>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine binning range
    let max_count = counts_by_reason
        .values()
        .flat_map(|v| v.iter())
        .copied()
        .max()
        .unwrap_or(0);
    if max_count == 0 {
        eprintln!("No accesses to plot");
        return Ok(());
    }

    // For each reason, count how many entries fell into each count 'c'
    let mut binned: HashMap<CacheAccessReason, HashMap<u32, u32>> = HashMap::new();
    for (&reason, vec) in counts_by_reason {
        let inner = binned.entry(reason).or_default();
        for &c in vec {
            *inner.entry(c).or_insert(0) += 1;
        }
    }

    // Compute total height per bin to set y-axis
    let mut total_per_bin: HashMap<u32, u32> = HashMap::new();
    for bin in 1..=max_count {
        let sum: u32 = binned.values().map(|m| *m.get(&bin).unwrap_or(&0)).sum();
        total_per_bin.insert(bin, sum);
    }
    let y_max = *total_per_bin.values().max().unwrap_or(&0);

    // Prepare a palette of styles for each reason
    let reasons: Vec<_> = binned.keys().cloned().collect();
    let palette: Vec<ShapeStyle> = (0..reasons.len())
        .map(|i| ShapeStyle {
            color: Palette99::pick(i).to_rgba().into(),
            filled: true,
            stroke_width: 0,
        })
        .collect();

    // Set up the drawing area
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Entry-Access Histogram (stacked by reason)",
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..(max_count + 1), 0u32..(y_max + 1))?;

    chart
        .configure_mesh()
        .x_desc("Times Accessed")
        .y_desc("Number of Entries")
        .disable_mesh()
        .draw()?;

    // Draw stacked bars
    // Track running offsets per bin
    let mut offsets: HashMap<u32, u32> = HashMap::new();
    for (idx, reason) in reasons.iter().enumerate() {
        let style = &palette[idx];
        for (&bin, &count) in &binned[reason] {
            let y0 = *offsets.get(&bin).unwrap_or(&0);
            let y1 = y0 + count;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(bin, y0), (bin + 1, y1)],
                style.clone(),
            )))?;
            offsets.insert(bin, y1);
        }
    }

    // Add a legend
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}
