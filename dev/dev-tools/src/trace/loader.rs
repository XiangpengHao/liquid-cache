use dioxus::prelude::*;

/// Server function to list all snapshot files in the snapshots directory
#[server]
pub async fn list_snapshots() -> Result<Vec<String>, ServerFnError> {
    use std::fs;
    use std::path::PathBuf;
    let snapshots_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("src/storage/src/cache/tests/snapshots");
    
    let mut files = Vec::new();
    
    if let Ok(entries) = fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.ends_with(".snap") {
                    files.push(file_name.to_string());
                }
            }
        }
    }
    
    files.sort();
    Ok(files)
}

/// Server function to load a specific snapshot file
#[server]
pub async fn load_snapshot(filename: String) -> Result<String, ServerFnError> {
    use std::fs;
    use std::path::PathBuf;
    let snapshots_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("src/storage/src/cache/tests/snapshots");
    
    let file_path = snapshots_dir.join(&filename);
    
    // Security check: ensure the path is still within snapshots directory
    if !file_path.starts_with(&snapshots_dir) {
        return Err(ServerFnError::new("Invalid file path"));
    }
    
    fs::read_to_string(&file_path)
        .map_err(|e| ServerFnError::new(format!("Failed to read file: {}", e)))
}
