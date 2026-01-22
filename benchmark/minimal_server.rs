use clap::Parser;
use liquid_cache_benchmarks::{MinimalServerConfig, setup_observability, start_minimal_server};
use liquid_cache_common::IoMode;
use log::info;
use mimalloc::MiMalloc;
use std::{net::SocketAddr, path::PathBuf};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(name = "Minimal LiquidCache Server")]
struct CliArgs {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1:15214")]
    address: SocketAddr,

    /// Abort the server if any thread panics
    #[arg(long = "abort-on-panic")]
    abort_on_panic: bool,

    /// Enable Liquid Cache squeezing
    #[arg(long = "enable-squeeze", default_value_t = true)]
    enable_squeeze: bool,

    /// Maximum cache size in MB
    #[arg(long = "max-cache-mb")]
    max_cache_mb: Option<usize>,

    /// Path to disk cache directory
    #[arg(long = "disk-cache-dir")]
    disk_cache_dir: Option<PathBuf>,

    /// Jaeger OTLP gRPC endpoint (for example: http://localhost:4317)
    #[arg(long = "jaeger-endpoint")]
    jaeger_endpoint: Option<String>,

    /// IO mode, available options: uring, uring-direct, std-blocking, tokio, std-spawn-blocking
    #[arg(long = "io-mode", default_value = "uring-multi-async")]
    io_mode: IoMode,

    /// Enable parquet row-group pruning (zone mapping)
    #[arg(long = "zone-mapping", default_value_t = true)]
    zone_mapping: bool,

    /// Enable parquet filter pushdown
    #[arg(long = "filter-pushdown", default_value_t = true)]
    filter_pushdown: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    setup_observability("minimal-liquid-cache-server", args.jaeger_endpoint.as_deref());

    if args.abort_on_panic {
        std::panic::set_hook(Box::new(|info| {
            eprintln!("Some thread panicked: {info:?}");
            std::process::exit(1);
        }));
    }

    let _server = start_minimal_server(MinimalServerConfig {
        address: args.address,
        max_cache_mb: args.max_cache_mb,
        disk_cache_dir: args.disk_cache_dir.clone(),
        enable_squeeze: args.enable_squeeze,
        io_mode: args.io_mode,
        zone_mapping: args.zone_mapping,
        filter_pushdown: args.filter_pushdown,
    })?;

    info!("LiquidCache server listening on {}", args.address);

    std::future::pending::<()>().await;
    Ok(())
}
