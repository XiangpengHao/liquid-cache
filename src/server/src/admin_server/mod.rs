//! Admin server for the liquid cache server
//!
//! This server is used to manage the liquid cache server

use axum::http::{HeaderValue, Method};
use axum::{
    Router,
    routing::{get, post},
};
use flamegraph::FlameGraph;
use std::sync::atomic::AtomicU32;
use std::{net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

mod flamegraph;
mod handlers;
mod disk_monitor;
pub(crate) mod models;

use crate::admin_server::disk_monitor::DiskMonitor;
use crate::LiquidCacheService;

pub(crate) struct AppState {
    liquid_cache: Arc<LiquidCacheService>,
    trace_id: AtomicU32,
    stats_id: AtomicU32,
    flamegraph: Arc<FlameGraph>,
    disk_monitor: Arc<DiskMonitor>,
}

/// Run the admin server
pub async fn run_admin_server(
    addr: SocketAddr,
    liquid_cache: Arc<LiquidCacheService>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        liquid_cache,
        trace_id: AtomicU32::new(0),
        stats_id: AtomicU32::new(0),
        flamegraph: Arc::new(FlameGraph::new()),
        disk_monitor: Arc::new(DiskMonitor::new()),
    });

    // Create a CORS layer that allows all localhost origins
    let cors = CorsLayer::new()
        // Allow all localhost origins (http and https)
        .allow_origin([
            "http://localhost:3000".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:3000".parse::<HeaderValue>().unwrap(),
            "https://liquid-cache-admin.xiangpeng.systems"
                .parse::<HeaderValue>()
                .unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/shutdown", get(handlers::shutdown_handler))
        .route("/reset_cache", get(handlers::reset_cache_handler))
        .route(
            "/parquet_cache_usage",
            get(handlers::get_parquet_cache_usage_handler),
        )
        .route("/cache_info", get(handlers::get_cache_info_handler))
        .route("/system_info", get(handlers::get_system_info_handler))
        .route("/start_trace", get(handlers::start_trace_handler))
        .route("/stop_trace", get(handlers::stop_trace_handler))
        .route(
            "/execution_metrics",
            get(handlers::get_execution_metrics_handler),
        )
        .route("/execution_plans", get(handlers::get_execution_stats))
        .route("/cache_stats", get(handlers::get_cache_stats_handler))
        .route("/start_flamegraph", get(handlers::start_flamegraph_handler))
        .route("/stop_flamegraph", get(handlers::stop_flamegraph_handler))
        .route(
            "/set_execution_stats",
            post(handlers::add_execution_stats_handler),
        )
        .route("/start_disk_usage_monitor", get(handlers::start_disk_usage_monitor_handler))
        .route("/stop_disk_usage_monitor", get(handlers::stop_disk_usage_monitor_handler))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
