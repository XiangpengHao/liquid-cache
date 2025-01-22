use std::sync::Arc;

use datafusion::{
    error::Result,
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::SplitSqlTableFactory;
use log::info;
use url::Url;

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder().format_timestamp(None).init();

    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let entry_point = "http://localhost:50051";

    let sql = "SELECT COUNT(*) FROM small_hits WHERE \"URL\" <> '';";

    info!("SQL to be executed: {}", sql);

    let table_name = "small_hits";

    let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
    let table_url = Url::parse(&format!(
        "file://{}/examples/small_hits.parquet",
        current_dir
    ))
    .unwrap();

    let table = SplitSqlTableFactory::open_table(entry_point, table_name, table_url).await?;
    ctx.register_table(table_name, Arc::new(table))?;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
