use arrow_flight::flight_service_server::FlightServiceServer;
use clap::Parser;
use fastrace_tonic::FastraceServerLayer;
use liquid_cache_benchmarks::{FlameGraphReport, StatsReport, setup_observability};
use liquid_cache_server::{LiquidCacheService, admin_server::run_admin_server};
use log::info;
use mimalloc::MiMalloc;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tonic::transport::Server;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(name = "ClickBench Benchmark Server")]
struct CliArgs {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1:50051")]
    address: SocketAddr,

    /// HTTP address for admin endpoint
    #[arg(long = "admin-address", default_value = "127.0.0.1:50052")]
    admin_address: SocketAddr,

    /// Abort the server if any thread panics
    #[arg(long = "abort-on-panic")]
    abort_on_panic: bool,

    /// Path to output flamegraph directory
    #[arg(long = "flamegraph-dir")]
    flamegraph_dir: Option<PathBuf>,

    /// Path to output cache internal stats directory
    #[arg(long = "stats-dir")]
    stats_dir: Option<PathBuf>,

    /// Maximum cache size in MB
    #[arg(long = "max-cache-mb")]
    max_cache_mb: Option<usize>,

    /// Path to disk cache directory
    #[arg(long = "disk-cache-dir")]
    disk_cache_dir: Option<PathBuf>,

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

    let ctx = LiquidCacheService::context()?;
    let mut liquid_cache_server =
        LiquidCacheService::new(ctx, max_cache_bytes, args.disk_cache_dir.clone());

    if let Some(flamegraph_dir) = &args.flamegraph_dir {
        assert!(
            flamegraph_dir.is_dir(),
            "Flamegraph output must be a directory"
        );
        liquid_cache_server
            .add_stats_collector(Arc::new(FlameGraphReport::new(flamegraph_dir.clone())));
    }

    if let Some(stats_dir) = &args.stats_dir {
        assert!(stats_dir.is_dir(), "Stats output must be a directory");
        liquid_cache_server.add_stats_collector(Arc::new(StatsReport::new(
            stats_dir.clone(),
            liquid_cache_server.cache().clone(),
        )));
    }

    let liquid_cache_server = Arc::new(liquid_cache_server);
    let flight = FlightServiceServer::from_arc(liquid_cache_server.clone());

    info!("LiquidCache server listening on {}", args.address);
    info!("Admin server listening on {}", args.admin_address);

    // Run both servers concurrently
    tokio::select! {
        result = Server::builder().layer(FastraceServerLayer).add_service(flight).serve(args.address) => {
            result?;
        },
        result = run_admin_server(args.admin_address, liquid_cache_server) => {
            result?;
        },
    }

    fastrace::flush();
    Ok(())
}
