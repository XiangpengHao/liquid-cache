use clap::Parser;
use liquid_cache_common::CacheAccessReason;
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
            color: Palette99::pick(i).to_rgba(),
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
        let style: &ShapeStyle = &palette[idx];
        for (&bin, &count) in &binned[reason] {
            let y0 = *offsets.get(&bin).unwrap_or(&0);
            let y1 = y0 + count;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(bin, y0), (bin + 1, y1)],
                *style,
            )))?;
            offsets.insert(bin, y1);
        }
    }

    // Add a legend
    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}
