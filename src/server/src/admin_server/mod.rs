//! Admin server for the liquid cache server
//!
//! This server is used to manage the liquid cache server

use axum::http::{HeaderValue, Method};
use axum::{
    Router,
    routing::{get, post},
};
use flamegraph::FlameGraph;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicU32;
use std::{net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

mod flamegraph;
mod handlers;

use crate::LiquidCacheService;

/// Response for the admin server
#[derive(Serialize, Deserialize)]
pub struct ApiResponse {
    /// Message for the response
    pub message: String,
    /// Status for the response
    pub status: String,
}

/// Parameters for the set_execution_stats endpoint
#[derive(Deserialize, Serialize, Clone)]
pub struct ExecutionStats {
    /// Plan ID for the execution plan
    pub plan_id: String,
    /// Display name for the execution plan
    pub display_name: String,
    /// Flamegraph SVG for the execution plan
    pub flamegraph_svg: Option<String>,
    /// Network traffic bytes for the execution plan
    pub network_traffic_bytes: u64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

pub(crate) struct AppState {
    liquid_cache: Arc<LiquidCacheService>,
    trace_id: AtomicU32,
    stats_id: AtomicU32,
    flamegraph: Arc<FlameGraph>,
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
        .route("/execution_plans", get(handlers::get_execution_plans))
        .route("/cache_stats", get(handlers::get_cache_stats_handler))
        .route("/start_flamegraph", get(handlers::start_flamegraph_handler))
        .route("/stop_flamegraph", get(handlers::stop_flamegraph_handler))
        .route(
            "/set_execution_stats",
            post(handlers::set_execution_stats_handler),
        )
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
