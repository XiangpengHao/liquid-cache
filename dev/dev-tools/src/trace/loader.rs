use dioxus::prelude::*;

/// Helper function to find all .snap files with EventTrace pattern
/// Returns absolute paths to matching files
#[cfg(any(feature = "server", test))]
pub fn find_event_trace_snapshots(src_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    use std::fs;
    use std::path::{Path, PathBuf};

    let mut matching_files = Vec::new();

    fn visit_dirs(dir: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, files);
                } else if let Some(file_name) = path.file_name().and_then(|n| n.to_str())
                    && file_name.ends_with(".snap")
                {
                    // Read the file and check for EventTrace pattern
                    if let Ok(content) = fs::read_to_string(&path)
                        && content.contains("EventTrace: [")
                    {
                        files.push(path);
                    }
                }
            }
        }
    }

    if src_dir.exists() {
        visit_dirs(src_dir, &mut matching_files);
    }

    // Sort by modification time (most recent first)
    matching_files.sort_by(|a, b| {
        let a_time = fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let b_time = fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        b_time.cmp(&a_time) // Reverse order: newest first
    });
    matching_files
}

/// Server function to list all snapshot files containing EventTrace pattern
/// Recursively scans the entire src/ tree for .snap files with EventTrace: [
#[server]
pub async fn list_snapshots() -> Result<Vec<String>, ServerFnError> {
    let src_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("src");

    let matching_files = find_event_trace_snapshots(&src_dir);

    // Convert to relative paths starting with "src/"
    let relative_paths: Vec<String> = matching_files
        .iter()
        .filter_map(|path| {
            path.strip_prefix(&src_dir)
                .ok()
                .map(|rel| format!("src/{}", rel.display()))
        })
        .collect();

    Ok(relative_paths)
}

/// Server function to load a specific snapshot file
#[server]
pub async fn load_snapshot(filename: String) -> Result<String, ServerFnError> {
    use std::fs;
    use std::path::PathBuf;

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let file_path = project_root.join(&filename);

    // Security check: ensure the path is within the project and in src/
    if !file_path.starts_with(&project_root) || !filename.starts_with("src/") {
        return Err(ServerFnError::new("Invalid file path"));
    }

    fs::read_to_string(&file_path)
        .map_err(|e| ServerFnError::new(format!("Failed to read file: {}", e)))
}
