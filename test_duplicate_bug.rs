use std::sync::Arc;
use std::fs;
use tempfile::TempDir;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTableUrl};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::aggregates::AggregateMode;
use datafusion::physical_plan::aggregates::PhysicalGroupBy;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::scalar::ScalarValue;
use datafusion::error::Result;
use arrow::array::{Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatchReader;
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
    
    // Test query: SELECT ___row_id FROM test WHERE target_status_code = 1 LIMIT 10000
    println!("Testing with liquid cache...");
    let liquid_result = liquid_ctx.sql("SELECT ___row_id FROM test_liquid WHERE target_status_code = 1 LIMIT 10000").await?;
    let liquid_batches = liquid_result.collect().await?;
    
    let mut liquid_row_ids = Vec::new();
    for batch in liquid_batches {
        let row_id_array = batch.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| datafusion::error::DataFusionError::Internal("Expected Int32Array".to_string()))?;
        
        liquid_row_ids.extend(row_id_array.iter().flatten());
    }
    
    println!("Liquid cache results: {} rows, {} duplicates", 
             liquid_row_ids.len(), 
             find_duplicates(&liquid_row_ids).len());
    
    println!("Testing without liquid cache...");
    let df_result = df_ctx.sql("SELECT ___row_id FROM test_df WHERE target_status_code = 1 LIMIT 10000").await?;
    let df_batches = df_result.collect().await?;
    
    let mut df_row_ids = Vec::new();
    for batch in df_batches {
        let row_id_array = batch.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| datafusion::error::DataFusionError::Internal("Expected Int32Array".to_string()))?;
        
        df_row_ids.extend(row_id_array.iter().flatten());
    }
    
    println!("DataFusion results: {} rows, {} duplicates", 
             df_row_ids.len(), 
             find_duplicates(&df_row_ids).len());
    
    // Compare results
    if find_duplicates(&liquid_row_ids).len() > 0 {
        println!("BUG REPRODUCED: Liquid cache produces duplicates!");
        println!("Liquid cache duplicates: {:?}", find_duplicates(&liquid_row_ids));
    } else {
        println!("No duplicates found in liquid cache results");
    }
    
    if find_duplicates(&df_row_ids).len() > 0 {
        println!("WARNING: DataFusion also produces duplicates!");
    }
    
    Ok(())
}