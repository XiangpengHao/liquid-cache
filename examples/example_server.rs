use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionContext;
use liquid_cache_server::LiquidCacheService;
use liquid_cache_server::common::CacheMode;
use liquid_cache_server::store::store::policies::LruPolicy;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),          // max memory cache size 1GB
        Some(tempfile::tempdir()?.keep()), // disk cache dir
        CacheMode::LiquidEagerTranscode,
        Box::new(LruPolicy::new()),
    )?;

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:15214".parse()?)
        .await?;

    Ok(())
}
