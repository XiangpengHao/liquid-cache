use arrow_flight::flight_service_server::FlightServiceServer;
use axum::Router;
use clap::Parser;
use fastrace_tonic::FastraceServerLayer;
use liquid_cache_benchmarks::setup_observability;
use liquid_cache_common::CacheMode;
use liquid_cache_store::store::policies::DiscardPolicy;
use liquid_cache_server::{LiquidCacheService, run_admin_server};
use log::info;
use mimalloc::MiMalloc;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::net::TcpListener;
use tonic::transport::Server;
use tower_http::services::ServeDir;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(name = "ClickBench Benchmark Server")]
struct CliArgs {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1:15214")]
    address: SocketAddr,

    /// HTTP address for admin endpoint
    #[arg(long = "admin-address", default_value = "127.0.0.1:53703")]
    admin_address: SocketAddr,

    /// Abort the server if any thread panics
    #[arg(long = "abort-on-panic")]
    abort_on_panic: bool,

    /// Maximum cache size in MB
    #[arg(long = "max-cache-mb")]
    max_cache_mb: Option<usize>,

    /// Path to disk cache directory
    #[arg(long = "disk-cache-dir")]
    disk_cache_dir: Option<PathBuf>,

    /// Cache mode
    #[arg(long = "cache-mode", default_value = "liquid_eager_transcode")]
    cache_mode: CacheMode,

    /// Static files directory (only used in static_file_server mode)
    #[arg(long = "static-dir", default_value = "static")]
    static_dir: PathBuf,

    /// Openobserve auth token
    #[arg(long)]
    openobserve_auth: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    setup_observability(
        "liquid-cache-server",
        opentelemetry::trace::SpanKind::Server,
        args.openobserve_auth.as_deref(),
    );

    let max_cache_bytes = args.max_cache_mb.map(|size| size * 1024 * 1024);

    if args.abort_on_panic {
        // Be loud and crash loudly if any thread panics.
        // This will stop the server if any thread panics.
        // But will prevent debugger to break on panic.
        std::panic::set_hook(Box::new(|info| {
            eprintln!("Some thread panicked: {info:?}");
            std::process::exit(1);
        }));
    }

    match args.cache_mode {
        CacheMode::StaticFileServer => {
            // Static file server mode
            let serve_dir = ServeDir::new(&args.static_dir);
            let app = Router::new().fallback_service(serve_dir);

            info!("Static file server listening on http://{}", args.address);
            info!("Serving files from directory: {:?}", args.static_dir);

            axum::serve(
                TcpListener::bind(args.address).await?,
                app.into_make_service(),
            )
            .await?;
        }
        _ => {
            // LiquidCache server mode
            let ctx = LiquidCacheService::context()?;
            let liquid_cache_server = LiquidCacheService::new(
                ctx,
                max_cache_bytes,
                args.disk_cache_dir.clone(),
                args.cache_mode,
                Box::new(DiscardPolicy),
            )?;

            let liquid_cache_server = Arc::new(liquid_cache_server);
            let flight = FlightServiceServer::from_arc(liquid_cache_server.clone());

            info!("LiquidCache server listening on {}", args.address);
            info!("Admin server listening on {}", args.admin_address);
            info!(
                "Dashboard: https://liquid-cache-admin.xiangpeng.systems/?host=http://{}",
                args.admin_address
            );

            // Run both servers concurrently
            tokio::select! {
                result = Server::builder().layer(FastraceServerLayer).add_service(flight).serve(args.address) => {
                    result?;
                },
                result = run_admin_server(args.admin_address, liquid_cache_server) => {
                    result?;
                },
            }
        }
    }

    fastrace::flush();
    Ok(())
}
