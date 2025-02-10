use arrow_flight::flight_service_server::FlightServiceServer;
use clap::{Command, arg, value_parser};
use liquid_cache_benchmarks::{FlameGraphReport, MetricsReport, StatsReport};
use liquid_cache_server::LiquidCacheService;
use log::info;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tonic::transport::Server;

use mimalloc::MiMalloc;

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
            arg!(--"metrics-dir" <PATH>)
                .required(false)
                .help("Path to output metrics directory")
                .value_parser(value_parser!(PathBuf)),
        )
        .get_matches();

    let flamegraph_dir = matches.get_one::<PathBuf>("flamegraph-dir").cloned();
    let stats_dir = matches.get_one::<PathBuf>("stats-dir").cloned();
    let metrics_dir = matches.get_one::<PathBuf>("metrics-dir").cloned();
    let abort_on_panic = matches.get_flag("abort-on-panic");
    if abort_on_panic {
        // Be loud and crash loudly if any thread panics.
        // This will stop the server if any thread panics.
        // But will prevent debugger to break on panic.
        std::panic::set_hook(Box::new(|info| {
            eprintln!("Some thread panicked: {:?}", info);
            std::process::exit(1);
        }));
    }

    let addr = matches.get_one::<SocketAddr>("address").unwrap();
    let partitions = matches.get_one::<usize>("partitions").cloned();

    let ctx = LiquidCacheService::context(partitions)?;
    let mut split_sql = LiquidCacheService::new_with_context(ctx);

    if let Some(flamegraph_dir) = flamegraph_dir {
        assert!(
            flamegraph_dir.is_dir(),
            "Flamegraph output must be a directory"
        );
        split_sql.add_stats_collector(Arc::new(FlameGraphReport::new(flamegraph_dir)));
    }

    if let Some(stats_dir) = stats_dir {
        assert!(stats_dir.is_dir(), "Stats output must be a directory");
        split_sql.add_stats_collector(Arc::new(StatsReport::new(
            stats_dir,
            split_sql.cache().clone(),
        )));
    }

    if let Some(metrics_dir) = metrics_dir {
        assert!(metrics_dir.is_dir(), "Metrics output must be a directory");
        split_sql.add_stats_collector(Arc::new(MetricsReport::new(
            metrics_dir,
            split_sql.cache().clone(),
        )));
    }
    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(*addr).await?;

    Ok(())
}
