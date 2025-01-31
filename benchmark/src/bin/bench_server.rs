use arrow_flight::flight_service_server::FlightServiceServer;
use clap::{Command, arg, value_parser};
use liquid_cache_benchmarks::{FlameGraphReport, StatsReport};
use liquid_cache_server::LiquidCacheService;
use liquid_parquet::{LiquidCache, LiquidCacheMode};
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
        .get_matches();

    let flamegraph_dir = matches.get_one::<PathBuf>("flamegraph-dir").cloned();
    let stats_dir = matches.get_one::<PathBuf>("stats-dir").cloned();
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

    let default_addr = "0.0.0.0:50051".parse().unwrap();
    let addr = matches
        .get_one::<SocketAddr>("address")
        .unwrap_or(&default_addr);
    let partitions = matches.get_one::<usize>("partitions").cloned();

    let ctx = LiquidCacheService::context(partitions)?;
    let batch_size = ctx.state().config().batch_size();
    let liquid_cache = Arc::new(LiquidCache::new(
        LiquidCacheMode::InMemoryLiquid,
        batch_size,
    ));
    let mut split_sql =
        LiquidCacheService::new_with_ctx_and_cache(Arc::new(ctx), liquid_cache.clone());

    if let Some(flamegraph_dir) = flamegraph_dir {
        assert!(
            flamegraph_dir.is_dir(),
            "Flamegraph output must be a directory"
        );
        split_sql.add_stats_collector(Arc::new(FlameGraphReport::new(flamegraph_dir)));
    }

    if let Some(stats_dir) = stats_dir {
        assert!(stats_dir.is_dir(), "Stats output must be a directory");
        split_sql.add_stats_collector(Arc::new(StatsReport::new(stats_dir, liquid_cache)));
    }

    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(*addr).await?;

    Ok(())
}
