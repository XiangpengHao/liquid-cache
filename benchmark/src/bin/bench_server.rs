use std::net::SocketAddr;

use arrow_flight::flight_service_server::FlightServiceServer;
use clap::{Command, arg, value_parser};
use liquid_cache_server::{LiquidCacheConfig, LiquidCacheService};
use log::info;
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
            arg!(--"poison-panic")
                .required(false)
                .help("Poison panics")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let poison_panic = matches.get_flag("poison-panic");
    if poison_panic {
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
    let config = LiquidCacheConfig::default();
    let split_sql = LiquidCacheService::new_with_context_and_config(ctx, config);
    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(*addr).await?;

    Ok(())
}
