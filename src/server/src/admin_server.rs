//! Admin server for the liquid cache server
//!
//! This server is used to manage the liquid cache server

use axum::http::{HeaderValue, Method};
use axum::{
    Json, Router,
    extract::{Query, State},
    routing::get,
};
use liquid_cache_common::CacheMode;
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use log::info;
use serde::Serialize;
use std::sync::atomic::AtomicU32;
use std::{collections::HashMap, fs, net::SocketAddr, path::Path, sync::Arc};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::LiquidCacheService;

#[derive(Serialize)]
struct ApiResponse {
    message: String,
    status: String,
}

#[derive(Serialize)]
struct TableInfo {
    name: String,
    path: String,
    cache_mode: String,
}

#[derive(Serialize)]
struct TablesResponse {
    tables: Vec<TableInfo>,
    status: String,
}

#[derive(Serialize)]
struct ParquetCacheUsage {
    directory: String,
    file_count: usize,
    total_size_bytes: u64,
    status: String,
}

#[derive(Serialize)]
struct CacheInfo {
    batch_size: usize,
    max_cache_bytes: u64, // here we need to be u64 because wasm is 32 bit usize.
    memory_usage_bytes: u64,
    disk_usage_bytes: u64,
}

#[derive(serde::Deserialize)]
struct TraceParams {
    path: String,
}

#[derive(serde::Deserialize)]
struct ExecutionMetricsParams {
    plan_id: String,
}

async fn shutdown_handler() -> Json<ApiResponse> {
    info!("Shutdown request received, shutting down server...");

    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });

    Json(ApiResponse {
        message: "Server shutting down...".to_string(),
        status: "success".to_string(),
    })
}

async fn reset_cache_handler(State(state): State<Arc<AppState>>) -> Json<ApiResponse> {
    info!("Resetting cache...");
    state.liquid_cache.cache().reset();

    Json(ApiResponse {
        message: "Cache reset successfully".to_string(),
        status: "success".to_string(),
    })
}

fn get_registered_tables_inner(tables: HashMap<String, (String, CacheMode)>) -> Vec<TableInfo> {
    tables
        .into_iter()
        .map(|(name, (path, mode))| TableInfo {
            name,
            path,
            cache_mode: mode.to_string(),
        })
        .collect()
}

async fn get_registered_tables_handler(State(state): State<Arc<AppState>>) -> Json<TablesResponse> {
    info!("Listing registered tables...");
    let tables = state.liquid_cache.get_registered_tables().await;
    let table_infos = get_registered_tables_inner(tables);
    Json(TablesResponse {
        tables: table_infos,
        status: "success".to_string(),
    })
}

fn get_parquet_cache_usage_inner(cache_dir: &Path) -> ParquetCacheUsage {
    let mut file_count = 0;
    let mut total_size: u64 = 0;

    fn walk_dir(dir: &Path) -> Result<(usize, u64), std::io::Error> {
        let mut count = 0;
        let mut size = 0;

        if dir.exists() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    count += 1;
                    let metadata = fs::metadata(&path)?;
                    size += metadata.len();
                } else if path.is_dir() {
                    let (sub_count, sub_size) = walk_dir(&path)?;
                    count += sub_count;
                    size += sub_size;
                }
            }
        }

        Ok((count, size))
    }

    if let Ok((count, size)) = walk_dir(cache_dir) {
        file_count = count;
        total_size = size;
    }

    ParquetCacheUsage {
        directory: cache_dir.to_string_lossy().to_string(),
        file_count,
        total_size_bytes: total_size,
        status: "success".to_string(),
    }
}

async fn get_parquet_cache_usage_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ParquetCacheUsage> {
    info!("Getting parquet cache usage...");
    let cache_dir = state.liquid_cache.get_parquet_cache_dir();
    let usage = get_parquet_cache_usage_inner(cache_dir);
    Json(usage)
}

async fn get_cache_info_handler(State(state): State<Arc<AppState>>) -> Json<CacheInfo> {
    info!("Getting cache info...");
    let batch_size = state.liquid_cache.cache().batch_size();
    let max_cache_bytes = state.liquid_cache.cache().max_cache_bytes() as u64;
    let memory_usage_bytes = state.liquid_cache.cache().memory_usage_bytes() as u64;
    let disk_usage_bytes = state.liquid_cache.cache().disk_usage_bytes() as u64;
    Json(CacheInfo {
        batch_size,
        max_cache_bytes,
        memory_usage_bytes,
        disk_usage_bytes,
    })
}

#[derive(Serialize)]
struct SystemInfo {
    total_memory_bytes: u64,
    used_memory_bytes: u64,
    available_memory_bytes: u64,
    name: String,
    kernel: String,
    os: String,
    host_name: String,
    cpu_cores: usize,
}

async fn get_system_info_handler(State(_state): State<Arc<AppState>>) -> Json<SystemInfo> {
    info!("Getting system info...");
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    Json(SystemInfo {
        total_memory_bytes: sys.total_memory(),
        used_memory_bytes: sys.used_memory(),
        available_memory_bytes: sys.available_memory(),
        name: sysinfo::System::name().unwrap_or_default(),
        kernel: sysinfo::System::kernel_version().unwrap_or_default(),
        os: sysinfo::System::os_version().unwrap_or_default(),
        host_name: sysinfo::System::host_name().unwrap_or_default(),
        cpu_cores: sysinfo::System::physical_core_count().unwrap_or(0),
    })
}

async fn start_trace_handler(State(state): State<Arc<AppState>>) -> Json<ApiResponse> {
    info!("Starting cache trace collection...");
    state.liquid_cache.cache().enable_trace();

    Json(ApiResponse {
        message: "Cache trace collection started".to_string(),
        status: "success".to_string(),
    })
}

async fn stop_trace_handler(
    Query(params): Query<TraceParams>,
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    info!("Stopping cache trace collection...");
    let save_path = Path::new(&params.path);

    match save_trace_to_file(save_path, &state) {
        Ok(_) => Json(ApiResponse {
            message: format!(
                "Cache trace collection stopped, saved to {}",
                save_path.display()
            ),
            status: "success".to_string(),
        }),
        Err(e) => Json(ApiResponse {
            message: format!("Failed to save trace: {e}"),
            status: "error".to_string(),
        }),
    }
}

fn save_trace_to_file(save_dir: &Path, state: &AppState) -> Result<(), Box<dyn std::error::Error>> {
    let now = std::time::SystemTime::now();
    let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    let minute = (datetime.as_secs() / 60) % 60;
    let second = datetime.as_secs() % 60;
    let trace_id = state
        .trace_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let filename = format!("cache-trace-id{trace_id:02}-{minute:02}-{second:03}.parquet",);

    // Ensure directory exists
    if !save_dir.exists() {
        fs::create_dir_all(save_dir)?;
    }

    let file_path = save_dir.join(filename);
    state.liquid_cache.cache().disable_trace();
    state.liquid_cache.cache().flush_trace(&file_path);
    Ok(())
}

async fn get_execution_metrics_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecutionMetricsParams>,
) -> Json<Option<ExecutionMetricsResponse>> {
    let Ok(uuid) = Uuid::parse_str(&params.plan_id) else {
        return Json(None);
    };
    let metrics = state.liquid_cache.inner().get_metrics(&uuid);
    Json(metrics)
}

struct AppState {
    liquid_cache: Arc<LiquidCacheService>,
    trace_id: AtomicU32,
}

/// Run the admin server
pub async fn run_admin_server(
    addr: SocketAddr,
    liquid_cache: Arc<LiquidCacheService>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        liquid_cache,
        trace_id: AtomicU32::new(0),
    });

    // Create a CORS layer that allows all localhost origins
    let cors = CorsLayer::new()
        // Allow all localhost origins (http and https)
        .allow_origin([
            "http://localhost:8080".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:8080".parse::<HeaderValue>().unwrap(),
            "http://liquid-cache-admin.xiangpeng.systems"
                .parse::<HeaderValue>()
                .unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/shutdown", get(shutdown_handler))
        .route("/reset_cache", get(reset_cache_handler))
        .route("/get_registered_tables", get(get_registered_tables_handler))
        .route("/parquet_cache_usage", get(get_parquet_cache_usage_handler))
        .route("/cache_info", get(get_cache_info_handler))
        .route("/system_info", get(get_system_info_handler))
        .route("/start_trace", get(start_trace_handler))
        .route("/stop_trace", get(stop_trace_handler))
        .route("/execution_metrics", get(get_execution_metrics_handler))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_get_registered_tables_inner() {
        let mut tables = HashMap::new();
        tables.insert(
            "table1".to_string(),
            ("s3://bucket/path1".to_string(), CacheMode::Parquet),
        );
        tables.insert(
            "table2".to_string(),
            ("s3://bucket/path2".to_string(), CacheMode::Liquid),
        );
        tables.insert(
            "table3".to_string(),
            ("s3://bucket/path3".to_string(), CacheMode::Arrow),
        );

        let result = get_registered_tables_inner(tables);

        assert_eq!(result.len(), 3);

        let mut sorted_result = result;
        sorted_result.sort_by(|a, b| a.name.cmp(&b.name));

        assert_eq!(sorted_result[0].name, "table1");
        assert_eq!(sorted_result[0].path, "s3://bucket/path1");
        assert_eq!(sorted_result[0].cache_mode, "parquet");

        assert_eq!(sorted_result[1].name, "table2");
        assert_eq!(sorted_result[1].path, "s3://bucket/path2");
        assert_eq!(sorted_result[1].cache_mode, "liquid");

        assert_eq!(sorted_result[2].name, "table3");
        assert_eq!(sorted_result[2].path, "s3://bucket/path3");
        assert_eq!(sorted_result[2].cache_mode, "arrow");
    }

    #[test]
    fn test_get_parquet_cache_usage_inner() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        let file1_path = temp_path.join("file1.parquet");
        let file2_path = temp_path.join("file2.parquet");

        let subdir_path = temp_path.join("subdir");
        std::fs::create_dir(&subdir_path).unwrap();
        let file3_path = subdir_path.join("file3.parquet");

        let data1 = [1u8; 1000];
        let data2 = [2u8; 2000];
        let data3 = [3u8; 3000];

        let mut file1 = std::fs::File::create(&file1_path).unwrap();
        file1.write_all(&data1).unwrap();

        let mut file2 = std::fs::File::create(&file2_path).unwrap();
        file2.write_all(&data2).unwrap();

        let mut file3 = std::fs::File::create(&file3_path).unwrap();
        file3.write_all(&data3).unwrap();

        // Expected total size: 6000 bytes (1000 + 2000 + 3000)

        let result = get_parquet_cache_usage_inner(temp_path);
        assert_eq!(result.directory, temp_path.to_string_lossy().to_string());
        assert_eq!(result.file_count, 3);
        assert_eq!(result.total_size_bytes, 6000);
        assert_eq!(result.status, "success");
    }

    #[test]
    fn test_get_parquet_cache_usage_inner_empty_dir() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();
        let result = get_parquet_cache_usage_inner(temp_path);
        assert_eq!(result.file_count, 0);
        assert_eq!(result.total_size_bytes, 0);
    }

    #[test]
    fn test_get_parquet_cache_usage_inner_nonexistent_dir() {
        let nonexistent_path = PathBuf::from("/path/does/not/exist");
        let result = get_parquet_cache_usage_inner(&nonexistent_path);
        assert_eq!(result.file_count, 0);
        assert_eq!(result.total_size_bytes, 0);
    }
}
