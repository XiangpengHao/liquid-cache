use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionContext;
use liquid_cache_datafusion_local::storage::cache::AlwaysHydrate;
use liquid_cache_datafusion_local::storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache_datafusion_server::LiquidCacheService;
use liquid_cache_datafusion_server::storage::cache_policies::LruPolicy;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),          // max memory cache size 1GB
        Some(tempfile::tempdir()?.keep()), // disk cache dir
        Box::new(LruPolicy::new()),
        Box::new(TranscodeSqueezeEvict),
        Box::new(AlwaysHydrate::new()),
    )?;

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:15214".parse()?)
        .await?;

    Ok(())
}
