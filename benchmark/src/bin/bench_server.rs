use arrow_flight::flight_service_server::FlightServiceServer;
use liquid_cache_server::SplitSqlService;
use log::info;
use tonic::transport::Server;

use clap::{ArgAction, Command, arg, command, value_parser};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("SplitSQL Benchmark Server")
        .arg(
            arg!(--"query-path" <PATH>)
                .required(true)
                .help("Path to the query file")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .arg(
            arg!(--query <NUMBER>)
                .required(false)
                .help("Query number to run")
                .value_parser(value_parser!(u32)),
        )
        .get_matches();

    env_logger::builder().format_timestamp(None).init();

    let addr = "0.0.0.0:50051".parse()?;

    let split_sql = SplitSqlService::try_new()?;
    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(addr).await?;

    Ok(())
}
