use arrow_flight::flight_service_server::FlightServiceServer;
use liquid_cache_server::LiquidCacheService;
use log::info;
use tonic::transport::Server;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder().format_timestamp(None).init();

    // Be loud and crash loudly if any thread panics.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("Some thread panicked: {:?}", info);
        std::process::exit(1);
    }));

    let addr = "0.0.0.0:50051".parse()?;

    let split_sql = LiquidCacheService::try_new()?;
    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(addr).await?;

    Ok(())
}
