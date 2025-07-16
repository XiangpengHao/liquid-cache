use anyhow::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionContext;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use url::Url;

use crate::Query;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ObjectStoreConfig {
    /// Object store URL (e.g., "s3://bucket-name", "gs://bucket-name", "http://localhost:9000")
    pub url: String,
    pub options: HashMap<String, String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BenchmarkManifest {
    /// Name of the benchmark suite
    pub name: String,
    /// Description of the benchmark
    pub description: Option<String>,
    /// Data tables configuration: table name -> parquet file path
    pub tables: HashMap<String, String>,
    /// Array of SQL queries - each element can be a file path or inline SQL
    pub queries: Vec<String>,
    /// Optional object store configurations
    pub object_stores: Option<Vec<ObjectStoreConfig>>,
}

impl BenchmarkManifest {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            tables: HashMap::new(),
            queries: Vec::new(),
            object_stores: None,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn add_table<S: Into<String>>(mut self, name: S, path: S) -> Self {
        self.tables.insert(name.into(), path.into());
        self
    }

    pub fn add_query<S: Into<String>>(mut self, query: S) -> Self {
        self.queries.push(query.into());
        self
    }

    pub fn save_to_file(&self, path: &PathBuf) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    pub fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    pub fn get_object_store(&self) -> Option<Vec<(ObjectStoreUrl, Box<dyn ObjectStore>)>> {
        let mut object_stores = Vec::new();
        for config in self.object_stores.as_ref()? {
            let url = ObjectStoreUrl::parse(&config.url).ok()?;
            let (object_store, _) =
                object_store::parse_url_opts(url.as_ref(), &config.options).ok()?;
            object_stores.push((url, object_store));
        }

        Some(object_stores)
    }

    pub async fn register_tables(&self, ctx: &SessionContext) -> Result<()> {
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        for (table_name, table_path) in &self.tables {
            let table_url = if table_path.starts_with("s3://")
                || table_path.starts_with("http://")
                || table_path.starts_with("https://")
            {
                Url::parse(table_path).expect("Failed to parse table path")
            } else {
                Url::parse(&format!("file://{current_dir}/{table_path}"))
                    .expect("Failed to parse table path")
            };

            ctx.register_parquet(table_name, table_url, Default::default())
                .await?;
        }
        Ok(())
    }

    pub async fn register_object_stores(&self, ctx: &SessionContext) -> Result<()> {
        if let Some(object_stores) = self.get_object_store() {
            for (url, object_store) in object_stores {
                ctx.register_object_store(url.as_ref(), Arc::new(object_store));
            }
        }
        Ok(())
    }

    pub fn load_queries(&self, id_offset: u32) -> Vec<Query> {
        let mut queries = Vec::new();
        for (index, query) in self.queries.iter().enumerate() {
            let sql = if std::path::Path::new(query).exists() {
                let raw_string = std::fs::read_to_string(query).expect("Failed to read query file");
                raw_string.split(";").map(|s| s.to_string()).collect()
            } else {
                vec![query.clone()]
            };

            queries.push(Query {
                id: index as u32 + id_offset,
                statement: sql,
            });
        }
        queries
    }
}
