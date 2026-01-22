use anyhow::{Result, anyhow};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, DictionaryArray, FixedSizeBinaryArray, LargeBinaryArray,
        LargeStringArray, StringArray, StructArray,
    },
    datatypes::{
        ArrowDictionaryKeyType, DataType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type, TimeUnit,
    },
};
use clap::Parser;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, SessionStateBuilder, TaskContext};
use datafusion::logical_expr::ScalarUDF;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_benchmarks::{BenchmarkManifest, Query, run_query, setup_observability};
use liquid_cache_common::IoMode;
use liquid_cache_parquet::optimizers::{LineageOptimizer, LocalModeOptimizer};
use liquid_cache_parquet::{
    LiquidCacheParquet, LiquidCacheParquetRef, VariantGetUdf, VariantPretty, VariantToJsonUdf,
};
use liquid_cache_storage::cache::NoHydration;
use liquid_cache_storage::cache::squeeze_policies::{
    SqueezePolicy, TranscodeEvict, TranscodeSqueezeEvict,
};
use liquid_cache_storage::cache_policies::LiquidPolicy;
use log::info;
use mimalloc::MiMalloc;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use futures::Stream;
use std::{
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll},
    time::Duration,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Clone)]
#[command(name = "Data Pruning Local Runner")]
struct CliArgs {
    /// Benchmark manifest file
    #[arg(long, default_value = "benchmark/clickbench/manifest.json")]
    manifest: PathBuf,

    /// Query id to run (defaults to all queries)
    #[arg(long)]
    query: Option<u32>,

    /// Number of partitions to use
    #[arg(long)]
    partitions: Option<usize>,

    /// Maximum cache size in MB
    #[arg(long = "max-cache-mb")]
    max_cache_mb: Option<usize>,

    /// Path to disk cache directory
    #[arg(long = "cache-dir")]
    cache_dir: Option<PathBuf>,

    /// Jaeger OTLP gRPC endpoint (for example: http://localhost:4317)
    #[arg(long = "jaeger-endpoint")]
    jaeger_endpoint: Option<String>,

    /// IO mode, available options: uring, uring-direct, std-blocking, tokio, std-spawn-blocking
    #[arg(long = "io-mode", default_value = "uring-multi-async")]
    io_mode: IoMode,

    /// Print physical plan for each statement (shows dynamic filters)
    #[arg(long = "print-plan", default_value_t = false)]
    print_plan: bool,
}

struct LayerConfig {
    name: &'static str,
    zone_mapping: bool,
    filter_pushdown: bool,
    dynamic_filtering: bool,
    squeeze: bool,
}

struct SqueezeUsage {
    mem_before: u64,
    disk_before: u64,
    mem_after: u64,
    disk_after: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();
    setup_observability("data-pruning-local-runner", args.jaeger_endpoint.as_deref());

    let manifest = BenchmarkManifest::load_from_file(&args.manifest)?;
    let raw_data_bytes = raw_data_bytes_for_manifest_tables(&manifest)?;

    let layers = [
        LayerConfig {
            name: "modern-encodings",
            zone_mapping: false,
            filter_pushdown: false,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping",
            zone_mapping: true,
            filter_pushdown: false,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping+filter-pushdown",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping+filter-pushdown+dynamic-filtering",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: true,
            squeeze: false,
        },
        /*LayerConfig {
            name: "zone-mapping+filter-pushdown+dynamic-filtering+squeeze",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: true,
            squeeze: true,
        },*/
    ];

    let mut queries = manifest.load_queries(0);
    if let Some(query_id) = args.query {
        queries.retain(|query| query.id() == query_id);
        if queries.is_empty() {
            return Err(anyhow!("No query with id {query_id} found in manifest"));
        }
    }

    let header = layers
        .iter()
        .map(|layer| layer.name)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "query,raw_data_bytes,{},squeeze_mem_before_bytes,squeeze_disk_before_bytes,squeeze_mem_after_bytes,squeeze_disk_after_bytes",
        header
    );
    for query in &queries {
        eprintln!("Starting query {}", query.id());
        let mut values = Vec::with_capacity(layers.len());
        for layer in &layers {
            eprintln!("Running query {} layer {}", query.id(), layer.name);
            let cache_dir = layer_cache_dir(args.cache_dir.as_ref(), query.id(), layer.name)?;

            let (ctx, cache) = build_local_context(&args, layer, cache_dir)?;
            manifest.register_object_stores(&ctx).await?;
            manifest.register_tables(&ctx).await?;

            let (mem_before, disk_before) = (
                cache.memory_usage_bytes() as u64,
                cache.disk_usage_bytes() as u64,
            );
            if mem_before != 0 || disk_before != 0 {
                eprintln!(
                    "Warning: cache not empty before query {} layer {} (mem={}, disk={})",
                    query.id(),
                    layer.name,
                    mem_before,
                    disk_before
                );
            }

            let ctx = Arc::new(ctx);
            let mut total_batches = 0usize;
            let mut total_rows = 0usize;
            let mut engine_bytes = 0u64;
            for statement in query.statement() {
                let (results, statement_bytes) =
                    run_query_with_engine_bytes(&ctx, statement, args.print_plan).await?;
                engine_bytes = engine_bytes.saturating_add(statement_bytes);
                total_batches += results.len();
                total_rows += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
            }

            let (mem_after, disk_after) = (
                cache.memory_usage_bytes() as u64,
                cache.disk_usage_bytes() as u64,
            );
            info!(
                "Layer {}, query {}: engine_bytes={}, cache_mem_bytes={}, cache_disk_bytes={}, batches={}, rows={}",
                layer.name,
                query.id(),
                engine_bytes,
                mem_after,
                disk_after,
                total_batches,
                total_rows
            );
            values.push(engine_bytes);

            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        eprintln!("Running query {} squeeze usage", query.id());
        let squeeze = run_squeeze_usage_for_query(&args, &manifest, query).await?;

        let values = values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{},{},{},{},{},{},{}",
            query.id(),
            raw_data_bytes,
            values,
            squeeze.mem_before,
            squeeze.disk_before,
            squeeze.mem_after,
            squeeze.disk_after
        );
    }
    Ok(())
}

async fn run_query_with_engine_bytes(
    ctx: &Arc<SessionContext>,
    query: &str,
    print_plan: bool,
) -> Result<(Vec<datafusion::arrow::array::RecordBatch>, u64)> {
    let df = ctx.sql(query).await?;
    let (state, logical_plan) = df.into_parts();
    let physical_plan = state.create_physical_plan(&logical_plan).await?;

    let counter = Arc::new(AtomicU64::new(0));
    let instrumented_plan = wrap_scan_nodes(physical_plan, counter.clone())?;
    if print_plan {
        eprintln!(
            "Physical plan:\n{}",
            datafusion::physical_plan::display::DisplayableExecutionPlan::new(
                instrumented_plan.as_ref()
            )
            .indent(true)
        );
    }

    let ctx = TaskContext::from(&state);
    let cfg = ctx
        .session_config()
        .clone()
        .with_extension(Arc::new(fastrace::Span::enter_with_local_parent(
            "poll_physical_plan",
        )));

    let execution_span = fastrace::Span::enter_with_local_parent("poll");
    let instrumented_plan = liquid_cache_benchmarks::instrument_liquid_source_with_span(
        instrumented_plan,
        execution_span,
    );

    let ctx = ctx.with_session_config(cfg);
    let results = collect(instrumented_plan.clone(), Arc::new(ctx)).await?;
    Ok((results, counter.load(Ordering::Relaxed)))
}

fn wrap_scan_nodes(
    plan: Arc<dyn ExecutionPlan>,
    counter: Arc<AtomicU64>,
) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
    if plan
        .as_any()
        .downcast_ref::<DataSourceExec>()
        .is_some()
        || plan.name() == "DataSourceExec"
    {
        return Ok(Arc::new(CountingExec::new(plan, counter)));
    }

    let mut new_children = Vec::with_capacity(plan.children().len());
    let mut children_changed = false;
    for child in plan.children() {
        let new_child = wrap_scan_nodes(child.clone(), counter.clone())?;
        if !Arc::ptr_eq(child, &new_child) {
            children_changed = true;
        }
        new_children.push(new_child);
    }

    if children_changed {
        plan.with_new_children(new_children)
    } else {
        Ok(plan)
    }
}

struct CountingExec {
    inner: Arc<dyn ExecutionPlan>,
    counter: Arc<AtomicU64>,
}

impl CountingExec {
    fn new(inner: Arc<dyn ExecutionPlan>, counter: Arc<AtomicU64>) -> Self {
        Self { inner, counter }
    }
}

impl std::fmt::Debug for CountingExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingExec")
    }
}

impl DisplayAs for CountingExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingExec")
    }
}

impl ExecutionPlan for CountingExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "CountingExec"
    }

    fn properties(&self) -> &PlanProperties {
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.inner]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            inner: children.first().unwrap().clone(),
            counter: self.counter.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let stream = self.inner.execute(partition, context)?;
        Ok(Box::pin(CountingStream::new(
            stream,
            self.counter.clone(),
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::common::Statistics> {
        self.inner.partition_statistics(None)
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.inner.supports_limit_pushdown()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        self.inner.with_fetch(limit)
    }

    fn fetch(&self) -> Option<usize> {
        self.inner.fetch()
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner.metrics()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.inner.maintains_input_order()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.inner.benefits_from_input_partitioning()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        self.inner.cardinality_effect()
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn datafusion::physical_plan::PhysicalExpr>>,
        config: &datafusion::config::ConfigOptions,
    ) -> datafusion::error::Result<FilterDescription> {
        self.inner
            .gather_filters_for_pushdown(phase, parent_filters, config)
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        config: &datafusion::config::ConfigOptions,
    ) -> datafusion::error::Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        self.inner
            .handle_child_pushdown_result(phase, child_pushdown_result, config)
    }
}

struct CountingStream {
    inner: SendableRecordBatchStream,
    counter: Arc<AtomicU64>,
}

impl CountingStream {
    fn new(inner: SendableRecordBatchStream, counter: Arc<AtomicU64>) -> Self {
        Self { inner, counter }
    }
}

impl Stream for CountingStream {
    type Item = datafusion::error::Result<datafusion::arrow::array::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let bytes = logical_value_bytes_for_batch(&batch);
                this.counter.fetch_add(bytes, Ordering::Relaxed);
                Poll::Ready(Some(Ok(batch)))
            }
            other => other,
        }
    }
}

impl RecordBatchStream for CountingStream {
    fn schema(&self) -> datafusion::arrow::datatypes::SchemaRef {
        self.inner.schema()
    }
}

fn build_local_context(
    args: &CliArgs,
    layer: &LayerConfig,
    cache_dir: PathBuf,
) -> Result<(SessionContext, LiquidCacheParquetRef)> {
    let mut session_config = SessionConfig::from_env()?;
    if let Some(partitions) = args.partitions {
        session_config.options_mut().execution.target_partitions = partitions;
    }

    let options = session_config.options_mut();
    options.execution.parquet.pruning = layer.zone_mapping;
    options.execution.parquet.pushdown_filters = layer.filter_pushdown;
    options.optimizer.enable_dynamic_filter_pushdown = layer.dynamic_filtering;
    options.execution.parquet.schema_force_view_types = false;
    options.execution.parquet.binary_as_string = true;
    options.execution.batch_size = 8192 * 2;

    let max_cache_bytes = args.max_cache_mb.map(|size| size * 1024 * 1024).unwrap_or(usize::MAX);

    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }
    std::fs::create_dir_all(&cache_dir)?;

    let squeeze_policy: Box<dyn SqueezePolicy> = if layer.squeeze {
        Box::new(TranscodeSqueezeEvict)
    } else {
        Box::new(TranscodeEvict)
    };

    let cache = LiquidCacheParquet::new(
        8192 * 2,
        max_cache_bytes,
        cache_dir,
        Box::new(LiquidPolicy::new()),
        squeeze_policy,
        Box::new(NoHydration::new()),
        args.io_mode,
    );
    let cache_ref = Arc::new(cache);

    let lineage_optimizer = Arc::new(LineageOptimizer::new());
    let optimizer = LocalModeOptimizer::new(cache_ref.clone(), true);

    let state = SessionStateBuilder::new()
        .with_config(session_config)
        .with_default_features()
        .with_optimizer_rule(lineage_optimizer)
        .with_physical_optimizer_rule(Arc::new(optimizer))
        .build();

    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));

    Ok((ctx, cache_ref))
}

async fn run_squeeze_usage_for_query(
    args: &CliArgs,
    manifest: &BenchmarkManifest,
    query: &Query,
) -> Result<SqueezeUsage> {
    let cache_dir = layer_cache_dir(args.cache_dir.as_ref(), query.id(), "squeeze")?;
    let mut session_config = SessionConfig::from_env()?;
    if let Some(partitions) = args.partitions {
        session_config.options_mut().execution.target_partitions = partitions;
    }

    let options = session_config.options_mut();
    options.execution.parquet.pruning = true;
    options.execution.parquet.pushdown_filters = true;
    options.optimizer.enable_dynamic_filter_pushdown = true;
    options.execution.parquet.schema_force_view_types = false;
    options.execution.parquet.binary_as_string = true;
    options.execution.batch_size = 8192 * 2;

    let max_cache_bytes = args.max_cache_mb.map(|size| size * 1024 * 1024).unwrap_or(usize::MAX);

    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }
    std::fs::create_dir_all(&cache_dir)?;

    let cache = LiquidCacheParquet::new(
        8192 * 2,
        max_cache_bytes,
        cache_dir,
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeSqueezeEvict),
        Box::new(NoHydration::new()),
        args.io_mode,
    );
    let cache_ref = Arc::new(cache);

    let lineage_optimizer = Arc::new(LineageOptimizer::new());
    let optimizer = LocalModeOptimizer::new(cache_ref.clone(), true);

    let state = SessionStateBuilder::new()
        .with_config(session_config)
        .with_default_features()
        .with_optimizer_rule(lineage_optimizer)
        .with_physical_optimizer_rule(Arc::new(optimizer))
        .build();

    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));

    manifest.register_object_stores(&ctx).await?;
    manifest.register_tables(&ctx).await?;

    let (mem_before, disk_before) = (
        cache_ref.memory_usage_bytes() as u64,
        cache_ref.disk_usage_bytes() as u64,
    );
    if mem_before != 0 || disk_before != 0 {
        eprintln!(
            "Warning: cache not empty before squeeze query {} (mem={}, disk={})",
            query.id(),
            mem_before,
            disk_before
        );
    }

    let ctx = Arc::new(ctx);
    for statement in query.statement() {
        let _ = run_query(&ctx, statement).await;
    }

    let (mem_before, disk_before) = (
        cache_ref.memory_usage_bytes() as u64,
        cache_ref.disk_usage_bytes() as u64,
    );
    cache_ref.squeeze_all_entries().await;
    let (mem_after, disk_after) = (
        cache_ref.memory_usage_bytes() as u64,
        cache_ref.disk_usage_bytes() as u64,
    );

    Ok(SqueezeUsage {
        mem_before,
        disk_before,
        mem_after,
        disk_after,
    })
}

fn layer_cache_dir(
    base: Option<&PathBuf>,
    query_id: u32,
    layer_name: &str,
) -> Result<PathBuf> {
    let base_dir = base
        .cloned()
        .unwrap_or(std::env::current_dir()?.join("benchmark/data/cache"));
    let sanitized = layer_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    Ok(base_dir.join(format!("q{}_{}", query_id, sanitized)))
}

fn raw_data_bytes_for_manifest_tables(manifest: &BenchmarkManifest) -> Result<u64> {
    let mut total = 0u64;
    for table_path in manifest.tables.values() {
        let path = resolve_local_table_path(table_path)?;
        let metadata = std::fs::metadata(&path)?;
        if metadata.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
                    total += raw_data_bytes_for_parquet(&entry_path)?;
                }
            }
        } else {
            total += raw_data_bytes_for_parquet(&path)?;
        }
    }
    Ok(total)
}

fn resolve_local_table_path(table_path: &str) -> Result<PathBuf> {
    if table_path.starts_with("s3://")
        || table_path.starts_with("http://")
        || table_path.starts_with("https://")
    {
        return Err(anyhow!(
            "raw data size only supports local files, got {table_path}"
        ));
    }
    if let Some(stripped) = table_path.strip_prefix("file://") {
        return Ok(PathBuf::from(stripped));
    }
    let current_dir = std::env::current_dir()?;
    Ok(current_dir.join(table_path))
}

fn raw_data_bytes_for_parquet(path: &Path) -> Result<u64> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let total_rows = builder.metadata().file_metadata().num_rows() as u64;
    let mut reader = builder.build()?;
    let mut sample_rows = 0u64;
    let mut sample_bytes = 0u64;
    for _ in 0..100 {
        let Some(batch) = reader.next() else {
            break;
        };
        let batch = batch?;
        sample_rows += batch.num_rows() as u64;
        sample_bytes += logical_value_bytes_for_batch(&batch);
    }
    if sample_rows == 0 {
        return Ok(0);
    }
    let avg_per_row = sample_bytes as f64 / sample_rows as f64;
    Ok((avg_per_row * total_rows as f64).round() as u64)
}

fn logical_value_bytes_for_batch(batch: &arrow::record_batch::RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|column| logical_value_bytes(column))
        .sum()
}

fn logical_value_bytes(array: &ArrayRef) -> u64 {
    let valid_len = (array.len() - array.null_count()) as u64;
    match array.data_type() {
        DataType::Boolean => (valid_len + 7) / 8,
        DataType::Int8 => valid_len * byte_width::<Int8Type>(),
        DataType::UInt8 => valid_len * byte_width::<UInt8Type>(),
        DataType::Int16 => valid_len * byte_width::<Int16Type>(),
        DataType::UInt16 => valid_len * byte_width::<UInt16Type>(),
        DataType::Int32 => valid_len * byte_width::<Int32Type>(),
        DataType::UInt32 => valid_len * byte_width::<UInt32Type>(),
        DataType::Int64 => valid_len * byte_width::<Int64Type>(),
        DataType::UInt64 => valid_len * byte_width::<UInt64Type>(),
        DataType::Float16 => valid_len * byte_width::<Float16Type>(),
        DataType::Float32 => valid_len * byte_width::<Float32Type>(),
        DataType::Float64 => valid_len * byte_width::<Float64Type>(),
        DataType::Date32 => valid_len * byte_width::<Date32Type>(),
        DataType::Date64 => valid_len * byte_width::<Date64Type>(),
        DataType::Time32(TimeUnit::Second) | DataType::Time32(TimeUnit::Millisecond) => {
            valid_len * 4
        }
        DataType::Time64(TimeUnit::Microsecond) | DataType::Time64(TimeUnit::Nanosecond) => {
            valid_len * 8
        }
        DataType::Timestamp(_, _) => valid_len * 8,
        DataType::Decimal128(_, _) => valid_len * byte_width::<Decimal128Type>(),
        DataType::Decimal256(_, _) => valid_len * byte_width::<Decimal256Type>(),
        DataType::FixedSizeBinary(byte_len) => {
            let array = array.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..array.len() {
                if array.is_valid(i) {
                    total += *byte_len as u64;
                }
            }
            total
        }
        DataType::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            sum_binary_lengths(array)
        }
        DataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            sum_large_binary_lengths(array)
        }
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<StringArray>().unwrap();
            sum_string_lengths(array)
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            sum_large_string_lengths(array)
        }
        DataType::Dictionary(key_type, value_type) => {
            match &**key_type {
                DataType::Int8 => dictionary_value_bytes::<Int8Type>(array, value_type),
                DataType::Int16 => dictionary_value_bytes::<Int16Type>(array, value_type),
                DataType::Int32 => dictionary_value_bytes::<Int32Type>(array, value_type),
                DataType::Int64 => dictionary_value_bytes::<Int64Type>(array, value_type),
                DataType::UInt8 => dictionary_value_bytes::<UInt8Type>(array, value_type),
                DataType::UInt16 => dictionary_value_bytes::<UInt16Type>(array, value_type),
                DataType::UInt32 => dictionary_value_bytes::<UInt32Type>(array, value_type),
                DataType::UInt64 => dictionary_value_bytes::<UInt64Type>(array, value_type),
                _ => 0,
            }
        }
        DataType::Struct(fields) => {
            fields
                .iter()
                .enumerate()
                .map(|(index, _field)| {
                    let child = array.as_any().downcast_ref::<StructArray>().unwrap();
                    logical_value_bytes(child.column(index))
                })
                .sum()
        }
        _ => 0,
    }
}

fn byte_width<T: arrow::datatypes::ArrowPrimitiveType>() -> u64 {
    std::mem::size_of::<T::Native>() as u64
}

fn sum_string_lengths(array: &StringArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_large_string_lengths(array: &LargeStringArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_binary_lengths(array: &BinaryArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_large_binary_lengths(array: &LargeBinaryArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn dictionary_value_bytes<K: arrow::datatypes::ArrowPrimitiveType + ArrowDictionaryKeyType>(
    array: &ArrayRef,
    value_type: &DataType,
) -> u64 {
    let dict = array
        .as_any()
        .downcast_ref::<DictionaryArray<K>>()
        .unwrap();
    let values = dict.values();
    let valid_len = (dict.len() - dict.null_count()) as u64;
    match value_type {
        DataType::Utf8 => {
            let values = values.as_any().downcast_ref::<StringArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::LargeUtf8 => {
            let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::Binary => {
            let values = values.as_any().downcast_ref::<BinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::LargeBinary => {
            let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::FixedSizeBinary(byte_len) => {
            valid_len * (*byte_len as u64)
        }
        DataType::Int8 => valid_len * byte_width::<Int8Type>(),
        DataType::UInt8 => valid_len * byte_width::<UInt8Type>(),
        DataType::Int16 => valid_len * byte_width::<Int16Type>(),
        DataType::UInt16 => valid_len * byte_width::<UInt16Type>(),
        DataType::Int32 => valid_len * byte_width::<Int32Type>(),
        DataType::UInt32 => valid_len * byte_width::<UInt32Type>(),
        DataType::Int64 => valid_len * byte_width::<Int64Type>(),
        DataType::UInt64 => valid_len * byte_width::<UInt64Type>(),
        DataType::Float16 => valid_len * byte_width::<Float16Type>(),
        DataType::Float32 => valid_len * byte_width::<Float32Type>(),
        DataType::Float64 => valid_len * byte_width::<Float64Type>(),
        DataType::Date32 => valid_len * byte_width::<Date32Type>(),
        DataType::Date64 => valid_len * byte_width::<Date64Type>(),
        DataType::Decimal128(_, _) => valid_len * byte_width::<Decimal128Type>(),
        DataType::Decimal256(_, _) => valid_len * byte_width::<Decimal256Type>(),
        _ => 0,
    }
}
