# DataFusion Cache (aka. SplitSQL)

You want a cache for object store, so we build this.

Most advanced caching system tailored for DataFusion.

## Try in 5 minutes!
Checkout the `examples` folder for more details.

### 1. Add dependency
```toml
[dependencies]
datafusion-cache = "0.1.0"
```

### 2. Start a cache server:
```rust
use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion_cache::cache::SplitSqlService;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse()?;

    let split_sql = SplitSqlService::try_new()?;

    Server::builder()
        .add_service(FlightServiceServer::new(split_sql))
        .serve(addr)
        .await?;

    Ok(())
}
```

### 3. Connect to the cache server:
```rust
use datafusion::prelude::*;
use datafusion_cache::compute::SplitSqlTableFactory;

#[tokio::main]
pub async fn main() -> Result<()> {
   let table = SplitSqlTableFactory::open_table(
        "http://localhost:50051",
        "table_name",
        "s3://bucket/your_file.parquet",
    )
    .await?;

    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("table_name", Arc::new(table))?;

    let sql = "SELECT COUNT(*) FROM small_hits WHERE \"URL\" <> '';";

    ctx.sql(sql).await?.show().await?;
    Ok(())
}
```

### 4. Enjoy!

## Development Setup


### (Optional) Setup


**Git Hooks**
After cloning the repository, run the following command to set up git hooks: 

```bash
./dev/install-git-hooks.sh
```

This will set up pre-commit hooks that check formatting, run clippy, and verify documentation.
