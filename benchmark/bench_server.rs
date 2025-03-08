use arrow_flight::flight_service_server::FlightServiceServer;
use clap::{Command, arg, value_parser};
use liquid_cache_benchmarks::{FlameGraphReport, StatsReport, admin_server::run_http_server};
use liquid_cache_server::LiquidCacheService;
use log::info;
use mimalloc::MiMalloc;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tonic::transport::Server;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder().format_timestamp(None).init();

    let matches = Command::new("ClickBench Benchmark Client")
        .arg(
            arg!(--"address" <ADDRESS>)
                .required(false)
                .default_value("127.0.0.1:50051")
                .help("Address to listen on")
                .value_parser(value_parser!(std::net::SocketAddr)),
        )
        .arg(
            arg!(--"admin-address" <ADDRESS>)
                .required(false)
                .default_value("127.0.0.1:8080")
                .help("HTTP address for admin endpoint")
                .value_parser(value_parser!(std::net::SocketAddr)),
        )
        .arg(
            arg!(--partitions<NUMBER>)
                .required(false)
                .help("Number of partitions to use")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            arg!(--"abort-on-panic")
                .required(false)
                .help("Abort the server if any thread panics")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            arg!(--"flamegraph-dir" <PATH>)
                .required(false)
                .help("Path to output flamegraph directory")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            arg!(--"stats-dir" <PATH>)
                .required(false)
                .help("Path to output cache internal stats directory")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            arg!(--"max-cache-mb" <SIZE>)
                .required(false)
                .help("Maximum cache size in MB")
                .value_parser(value_parser!(usize)),
        )
        .get_matches();

    let addr = matches.get_one::<SocketAddr>("address").unwrap();
    let admin_addr = matches.get_one::<SocketAddr>("admin-address").unwrap();
    let partitions = matches.get_one::<usize>("partitions").cloned();

    let flamegraph_dir = matches.get_one::<PathBuf>("flamegraph-dir").cloned();
    let stats_dir = matches.get_one::<PathBuf>("stats-dir").cloned();
    let abort_on_panic = matches.get_flag("abort-on-panic");
    let max_cache_bytes = matches
        .get_one::<usize>("max-cache-mb")
        .cloned()
        .map(|size| size * 1024 * 1024);
    if abort_on_panic {
        // Be loud and crash loudly if any thread panics.
        // This will stop the server if any thread panics.
        // But will prevent debugger to break on panic.
        std::panic::set_hook(Box::new(|info| {
            eprintln!("Some thread panicked: {:?}", info);
            std::process::exit(1);
        }));
    }

    let ctx = LiquidCacheService::context(partitions)?;
    let mut liquid_cache_server = LiquidCacheService::new_with_context(ctx, max_cache_bytes);

    if let Some(flamegraph_dir) = flamegraph_dir {
        assert!(
            flamegraph_dir.is_dir(),
            "Flamegraph output must be a directory"
        );
        liquid_cache_server.add_stats_collector(Arc::new(FlameGraphReport::new(flamegraph_dir)));
    }

    if let Some(stats_dir) = stats_dir {
        assert!(stats_dir.is_dir(), "Stats output must be a directory");
        liquid_cache_server.add_stats_collector(Arc::new(StatsReport::new(
            stats_dir,
            liquid_cache_server.cache().clone(),
        )));
    }

    let flight = FlightServiceServer::new(liquid_cache_server);

    info!("LiquidCache server listening on {addr}");
    info!("Admin server listening on {admin_addr}");

    // Run both servers concurrently
    tokio::select! {
        result = Server::builder().add_service(flight).serve(*addr) => {
            result?;
        },
        result = run_http_server(*admin_addr) => {
            result?;
        },
    }

    Ok(())
}
