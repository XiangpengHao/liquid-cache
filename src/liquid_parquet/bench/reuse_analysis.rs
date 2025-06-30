use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::Path;

fn pack_u16s(a: u16, b: u16, c: u16, d: u16) -> u64 {
    ((a as u64) << 48) | ((b as u64) << 32) | ((c as u64) << 16) | (d as u64)
}

fn analyze_trace_file(file_path: &Path) -> Vec<(u64, u64)> {
    let mut trace_data = Vec::new();
    let mut key_to_positions: HashMap<u64, Vec<usize>> = HashMap::new();
    
    // Read the parquet file
    let file = File::open(file_path).expect("Failed to open parquet file");
    let parquet_reader = SerializedFileReader::new(file).expect("Failed to create parquet reader");
    let row_iter = parquet_reader.get_row_iter(None).expect("Failed to get row iterator");

    // First pass: collect all positions for each key
    for (pos, row) in row_iter.enumerate() {
        let row = row.expect("Failed to read row");
        let file_id: u16 = row.get_ulong(0).expect("Failed to get file_id") as u16;
        let row_group_id: u16 = row.get_ulong(1).expect("Failed to get row_group_id") as u16;
        let column_id: u16 = row.get_ulong(2).expect("Failed to get column_id") as u16;
        let batch_id: u16 = row.get_ulong(3).expect("Failed to get batch_id") as u16;

        let key = pack_u16s(file_id, row_group_id, column_id, batch_id);
        trace_data.push(key);
        
        key_to_positions.entry(key).or_insert_with(Vec::new).push(pos);
    }

    // Second pass: calculate reuse distance for each access
    let mut reuse_distances = Vec::new();
    
    for (access_time, &key) in trace_data.iter().enumerate() {
        let positions = key_to_positions.get(&key).unwrap();
        
        // Find the next occurrence of this key after the current position
        let next_pos = positions.iter()
            .find(|&&pos| pos > access_time)
            .copied();
        
        let reuse_distance = match next_pos {
            Some(next_pos) => (next_pos - access_time) as u64,
            None => 0, // No future access
        };
        
        reuse_distances.push((access_time as u64, reuse_distance));
    }
    
    reuse_distances
}

fn main() {
    let trace_dir = Path::new("./trace");
    if !trace_dir.exists() {
        eprintln!("Trace directory './trace' does not exist!");
        std::process::exit(1);
    }

    let paths = fs::read_dir(trace_dir).expect("Failed to read trace directory");
    
    // Collect all trace files and their data
    let mut trace_files = Vec::new();
    let mut all_reuse_distances = Vec::new();
    let mut max_length = 0;

    for entry in paths {
        let entry = entry.expect("Failed to read directory entry");
        let file_path = entry.path();
        
        if file_path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            let file_name = file_path.file_name().unwrap().to_str().unwrap().to_string();
            
            let reuse_distances = analyze_trace_file(&file_path);
            let length = reuse_distances.len();
            
            trace_files.push(file_name);
            all_reuse_distances.push(reuse_distances);
            max_length = max_length.max(length);
        }
    }

    if trace_files.is_empty() {
        eprintln!("No parquet files found in trace directory!");
        std::process::exit(1);
    }

    // Print CSV header
    print!("access_time");
    for file_name in &trace_files {
        print!(",{}_reuse_distance", file_name);
    }
    println!();

    // Print data rows
    for access_time in 0..max_length {
        print!("{}", access_time);
        
        for reuse_distances in &all_reuse_distances {
            if access_time < reuse_distances.len() {
                print!(",{}", reuse_distances[access_time].1);
            } else {
                print!(","); // Empty cell for shorter traces
            }
        }
        println!();
    }
} 