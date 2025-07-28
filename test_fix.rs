use std::sync::Arc;
use std::fs;
use tempfile::TempDir;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTableUrl};
use datafusion::error::Result;
use arrow::array::{Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use liquid_cache_parquet::{
    LiquidCacheInProcessBuilder,
    common::LiquidCacheMode,
    cache::policies::DiscardPolicy,
};

fn create_test_parquet_file() -> Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let parquet_path = temp_dir.path().join("test.parquet");
    
    // Create test data with known row IDs
    let schema = Schema::new(vec![
        Field::new("target_status_code", DataType::Int32, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("___row_id", DataType::Int32, false),
    ]);
    
    // Create 10000 rows with unique row IDs
    let mut target_status_codes = Vec::new();
    let mut timestamps = Vec::new();
    let mut row_ids = Vec::new();
    
    for i in 0..10000 {
        target_status_codes.push(i % 5); // Some repeated values
        timestamps.push(i as i64);
        row_ids.push(i as i32); // Unique row IDs
    }
    
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(target_status_codes)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Int32Array::from(row_ids)),
        ],
    )?;
    
    // Write to parquet file
    let file = fs::File::create(&parquet_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    
    Ok(temp_dir)
}

fn find_duplicates<T: Eq + std::hash::Hash + Clone>(vec: &[T]) -> Vec<T> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    
    for item in vec {
        if !seen.insert(item.clone()) {
            duplicates.push(item.clone());
        }
    }
    
    duplicates
}

fn test_query_with_cache(ctx: &SessionContext, table_name: &str) -> Result<Vec<i32>> {
    let result = ctx.sql(&format!("SELECT ___row_id FROM {} WHERE target_status_code = 1 LIMIT 10000", table_name)).await?;
    let batches = result.collect().await?;
    
    let mut row_ids = Vec::new();
    for batch in batches {
        let row_id_array = batch.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| datafusion::error::DataFusionError::Internal("Expected Int32Array".to_string()))?;
        
        row_ids.extend(row_id_array.iter().flatten());
    }
    
    Ok(row_ids)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Creating test parquet file...");
    let temp_dir = create_test_parquet_file()?;
    let parquet_path = temp_dir.path().join("test.parquet");
    
    println!("Setting up liquid cache context...");
    let liquid_ctx = LiquidCacheInProcessBuilder::new()
        .with_max_cache_bytes(10 * 1024 * 1024 * 1024) // 10GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(LiquidCacheMode::Liquid {
            transcode_in_background: true,
        })
        .with_cache_strategy(Box::new(DiscardPolicy::default()))
        .build(SessionConfig::new())?;
    
    println!("Setting up regular DataFusion context...");
    let df_ctx = SessionContext::new();
    
    // Register the parquet file with both contexts
    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options = ListingOptions::new(Arc::new(file_format))
        .with_file_extension(".parquet");
    
    let table_path = ListingTableUrl::parse(parquet_path.to_str().unwrap())?;
    
    liquid_ctx.register_listing_table("test_liquid", &table_path, listing_options.clone(), None, None).await?;
    df_ctx.register_listing_table("test_df", &table_path, listing_options, None, None).await?;
    
    // Test multiple queries to trigger cache behavior
    println!("Running multiple queries to test cache behavior...");
    
    let mut liquid_results = Vec::new();
    let mut df_results = Vec::new();
    
    for i in 0..3 {
        println!("Query iteration {}", i + 1);
        
        let liquid_row_ids = test_query_with_cache(&liquid_ctx, "test_liquid").await?;
        let df_row_ids = test_query_with_cache(&df_ctx, "test_df").await?;
        
        liquid_results.push(liquid_row_ids);
        df_results.push(df_row_ids);
    }
    
    // Check for duplicates in each result
    for (i, (liquid_result, df_result)) in liquid_results.iter().zip(df_results.iter()).enumerate() {
        let liquid_duplicates = find_duplicates(liquid_result);
        let df_duplicates = find_duplicates(df_result);
        
        println!("Iteration {}: Liquid cache has {} duplicates, DataFusion has {} duplicates", 
                 i + 1, liquid_duplicates.len(), df_duplicates.len());
        
        if !liquid_duplicates.is_empty() {
            println!("  Liquid cache duplicates: {:?}", liquid_duplicates);
        }
        
        if !df_duplicates.is_empty() {
            println!("  DataFusion duplicates: {:?}", df_duplicates);
        }
    }
    
    // Check if results are consistent across iterations
    let first_liquid = &liquid_results[0];
    let first_df = &df_results[0];
    
    for (i, liquid_result) in liquid_results.iter().enumerate().skip(1) {
        if liquid_result != first_liquid {
            println!("WARNING: Liquid cache results are inconsistent across iterations!");
        }
        if &df_results[i] != first_df {
            println!("WARNING: DataFusion results are inconsistent across iterations!");
        }
    }
    
    println!("Test completed.");
    Ok(())
}