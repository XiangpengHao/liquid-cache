//! Admin server for the liquid cache server
//!
//! This server is used to manage the liquid cache server

use axum::{Router, extract::State, response::IntoResponse, routing::get};
use log::info;
use std::{net::SocketAddr, sync::Arc};

async fn shutdown_handler() -> impl IntoResponse {
    info!("Shutdown request received, shutting down server...");

    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });

    "Server shutting down..."
}

async fn reset_cache_handler(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    info!("Resetting cache...");
    "Cache reset..."
}

struct AppState {}

/// Run the admin server
pub async fn run_admin_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {});
    let app = Router::new()
        .route("/shutdown", get(shutdown_handler))
        .route("/reset_cache", get(reset_cache_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
