use arrow_flight::flight_service_server::FlightServiceServer;
use clap::{Parser, ValueEnum};
use datafusion::{error::Result, execution::object_store::ObjectStoreUrl, prelude::*};
use liquid_cache_datafusion_client::LiquidCacheClientBuilder;
use liquid_cache_datafusion_local::storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache_datafusion_local::storage::cache::{AlwaysHydrate, LiquidPolicy};
use liquid_cache_datafusion_server::LiquidCacheService;
use std::path::Path;
use std::sync::Arc;
use tonic::transport::Server;
use url::Url;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Server,
    Client,
}

#[derive(Parser, Clone)]
#[command(name = "Example Client/Server")]
struct CliArgs {
    /// Run as server or client
    #[arg(long, value_enum, default_value_t = Mode::Server)]
    mode: Mode,

    /// SQL query to execute (client mode)
    #[arg(
        long,
        default_value = "SELECT COUNT(*) FROM \"aws-edge-locations\" WHERE \"countryCode\" = 'US';"
    )]
    query: String,

    /// URL of the table to query (client mode)
    #[arg(
        long,
        default_value = "https://raw.githubusercontent.com/tobilg/aws-edge-locations/main/data/aws-edge-locations.parquet"
    )]
    file: String,

    /// Server address (host:port for server mode, URL for client mode)
    #[arg(long, default_value = "http://localhost:15214")]
    cache_server: String,
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    match args.mode {
        Mode::Server => run_server().await,
        Mode::Client => run_client(args).await.map_err(Into::into),
    }
}

async fn run_server() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),          // max memory size 1GB
        Some(tempfile::tempdir()?.keep()), // disk cache dir
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeSqueezeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await?;

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:15214".parse()?)
        .await?;

    Ok(())
}

async fn run_client(args: CliArgs) -> Result<()> {
    let url = Url::parse(&args.file).unwrap();
    let object_store_url = format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default());

    let ctx = LiquidCacheClientBuilder::new(args.cache_server.clone())
        .with_object_store(ObjectStoreUrl::parse(object_store_url.as_str())?, None)
        .build(SessionConfig::from_env()?)?;
    let ctx = Arc::new(ctx);

    let table_name = Path::new(url.path())
        .file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("default");
    let object_store = object_store::http::HttpBuilder::new()
        .with_url(object_store_url.as_str())
        .build()
        .unwrap();
    let object_store_url = ObjectStoreUrl::parse(object_store_url.as_str()).unwrap();
    ctx.register_object_store(object_store_url.as_ref(), Arc::new(object_store));
    ctx.register_parquet(table_name, url.as_ref(), Default::default())
        .await?;

    ctx.sql(&args.query).await?.show().await?;

    Ok(())
}
