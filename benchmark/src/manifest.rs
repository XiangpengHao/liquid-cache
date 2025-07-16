use anyhow::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

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
    /// Special handling for queries (e.g., TPC-H Q15 has multiple statements)
    pub special_query_handling: Option<HashMap<u32, String>>,
}

impl BenchmarkManifest {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            tables: HashMap::new(),
            queries: Vec::new(),
            object_stores: None,
            special_query_handling: None,
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

    pub fn with_special_query_handling(mut self, handling: HashMap<u32, String>) -> Self {
        self.special_query_handling = Some(handling);
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
}
