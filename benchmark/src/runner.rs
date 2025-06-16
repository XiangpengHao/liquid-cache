use crate::{
    BenchmarkResult, CommonBenchmarkArgs, IterationResult, Query, QueryResult, setup_observability,
};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    physical_plan::display::DisplayableExecutionPlan,
    prelude::SessionContext,
};
use fastrace::prelude::*;
use log::{debug, info};
use serde::Serialize;
use std::{fs::File, sync::Arc, time::Instant};
use sysinfo::{Disks, Networks};
use uuid::Uuid;

/// Trait that benchmarks must implement
#[allow(async_fn_in_trait)]
pub trait Benchmark: Serialize + Clone {
    type Args: Serialize + Clone;

    /// Get the common benchmark arguments
    fn common_args(&self) -> &CommonBenchmarkArgs;

    /// Get the benchmark-specific arguments
    fn args(&self) -> &Self::Args;

    /// Setup the session context for this benchmark
    async fn setup_context(&self) -> Result<Arc<SessionContext>>;

    /// Get all queries to run for this benchmark
    async fn get_queries(&self) -> Result<Vec<Query>>;

    /// Validate query results against expected answers (optional)
    async fn validate_result(&self, query: &Query, results: &[RecordBatch]) -> Result<()>;

    /// Custom query execution logic (optional, for special cases like TPCH Q15)
    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
    ) -> Result<(
        Vec<RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<Uuid>,
    )>;

    /// Get the benchmark name for tracing
    fn benchmark_name(&self) -> &'static str;
}

/// Generic benchmark runner that handles the common execution logic
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run a benchmark using the provided benchmark implementation
    pub async fn run<B: Benchmark>(benchmark: B) -> Result<BenchmarkResult<B::Args>> {
        let common = benchmark.common_args();

        setup_observability(
            benchmark.benchmark_name(),
            opentelemetry::trace::SpanKind::Client,
            common.openobserve_auth.as_deref(),
        );

        let ctx = benchmark.setup_context().await?;
        let queries = benchmark.get_queries().await?;
        let queries = if let Some(query) = common.query {
            vec![queries.into_iter().find(|q| q.id == query).unwrap()]
        } else {
            queries
        };

        let mut benchmark_result = BenchmarkResult {
            args: benchmark.args().clone(),
            results: Vec::new(),
        };

        let mut networks = Networks::new_with_refreshed_list();
        let mut disk = Disks::new_with_refreshed_list();
        let bench_start_time = Instant::now();

        for query in queries {
            let mut query_result = QueryResult::new(query.id, query.sql.clone());

            for it in 0..common.iteration {
                let iteration_result = Self::run_single_iteration(
                    &benchmark,
                    &ctx,
                    &query,
                    it,
                    &mut networks,
                    &mut disk,
                    bench_start_time,
                )
                .await?;

                query_result.add(iteration_result);
            }

            if common.reset_cache {
                common.reset_cache().await?;
            }

            benchmark_result.results.push(query_result);
        }

        if let Some(output_path) = &common.output {
            let output_file = File::create(output_path)?;
            serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
        }

        fastrace::flush();
        Ok(benchmark_result)
    }

    async fn run_single_iteration<B: Benchmark>(
        benchmark: &B,
        ctx: &Arc<SessionContext>,
        query: &Query,
        iteration: u32,
        networks: &mut Networks,
        disk: &mut Disks,
        bench_start_time: Instant,
    ) -> Result<IterationResult> {
        let common = benchmark.common_args();

        info!("Running query {}: \n{}", query.id, query.sql);

        common.start_trace().await;
        common.start_flamegraph().await;

        let root = Span::root(
            format!("{}-{}-{}", benchmark.benchmark_name(), query.id, iteration),
            SpanContext::random(),
        );
        let _g = root.set_local_parent();

        let now = Instant::now();
        let starting_timestamp = bench_start_time.elapsed();

        let (results, physical_plan, plan_uuid) = benchmark.execute_query(ctx, query).await?;
        let elapsed = now.elapsed();

        networks.refresh(true);
        let network_info = networks
            .get("lo0")
            .or_else(|| networks.get("lo"))
            .expect("No loopback interface found in networks");
        disk.refresh(true);
        let disk_read: u64 = disk.iter().map(|disk| disk.usage().read_bytes).sum();
        let disk_written: u64 = disk.iter().map(|disk| disk.usage().written_bytes).sum();

        let flamegraph = if !plan_uuid.is_empty() {
            common.stop_flamegraph().await
        } else {
            None
        };
        common.stop_trace().await;

        let physical_plan_with_metrics =
            DisplayableExecutionPlan::with_metrics(physical_plan.as_ref());
        debug!(
            "Physical plan: \n{}",
            physical_plan_with_metrics.indent(true)
        );
        let result_str = pretty::pretty_format_batches(&results).unwrap();
        debug!("Query result: \n{result_str}");

        benchmark.validate_result(query, &results).await?;

        common.get_cache_stats().await;
        let network_traffic = network_info.received();

        if !plan_uuid.is_empty() {
            common
                .set_execution_stats(
                    plan_uuid,
                    flamegraph,
                    format!("{}-q{}-{}", benchmark.benchmark_name(), query.id, iteration),
                    network_traffic,
                    elapsed.as_millis() as u64,
                    query.sql.clone(),
                )
                .await;
        }

        let metrics_response = common.get_execution_metrics(&physical_plan).await;

        let result = IterationResult {
            network_traffic,
            time_millis: elapsed.as_millis() as u64,
            cache_cpu_time: metrics_response.pushdown_eval_time,
            cache_memory_usage: metrics_response.cache_memory_usage,
            liquid_cache_usage: metrics_response.liquid_cache_usage,
            starting_timestamp,
            disk_bytes_read: disk_read,
            disk_bytes_written: disk_written,
        };

        info!("\n{result}");
        Ok(result)
    }
}
