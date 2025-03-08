use axum::{Router, response::IntoResponse, routing::get};
use log::info;
use std::net::SocketAddr;

async fn shutdown_handler() -> impl IntoResponse {
    info!("Shutdown request received, shutting down server...");

    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });

    "Server shutting down..."
}

pub async fn run_http_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new().route("/shutdown", get(shutdown_handler));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
