use std::{fmt, pin::Pin, sync::Arc};

use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Schema, SchemaRef};
use datafusion::{error::DataFusionError, physical_expr::projection::Projector};
use datafusion_datasource::morsel::Morsel;
use futures::{Stream, StreamExt, stream::BoxStream};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ArrowPredicate, RowSelection, RowSelector},
    },
    file::metadata::ParquetMetaData,
};

use crate::{
    cache::{CachedFileRef, CachedRowGroupRef, LiquidCacheParquetRef, RowGroupSnapshots},
    reader::plantime::{LiquidFileMetrics, LiquidRowFilter, ParquetMetadataCacheReader},
};

use super::{
    liquid_cache_reader::{
        LiquidCacheReader, LiquidCacheReaderConfig, ParquetFallback, ParquetFallbackConfig,
    },
    utils::get_root_column_ids,
};

pub(crate) struct LiquidRowGroupPlanner {
    pub(crate) metadata: Arc<ParquetMetaData>,
    pub(crate) input: ParquetMetadataCacheReader,
    pub(crate) row_filter: Option<LiquidRowFilter>,
    pub(crate) cached_file: CachedFileRef,
    pub(crate) projection: ProjectionMask,
    pub(crate) batch_size: usize,
    pub(crate) stream_schema: SchemaRef,
    pub(crate) output_schema: SchemaRef,
    pub(crate) projector: Arc<Projector>,
    pub(crate) replace_schema: bool,
    pub(crate) span: Option<Arc<fastrace::Span>>,
    pub(crate) liquid_cache: LiquidCacheParquetRef,
    pub(crate) metrics: LiquidFileMetrics,
}

impl LiquidRowGroupPlanner {
    fn cache_details(&self) -> CacheDetails {
        let schema_descr = self.metadata.file_metadata().schema_descr();
        let mut predicate_projection: Option<ProjectionMask> = None;
        if let Some(filter) = &self.row_filter {
            for predicate in filter.predicates() {
                let projection = predicate.projection();
                if let Some(predicate_projection) = &mut predicate_projection {
                    predicate_projection.union(projection);
                } else {
                    predicate_projection = Some(projection.clone());
                }
            }
        }
        let mut cache_projection = self.projection.clone();
        if let Some(predicate_projection) = &predicate_projection {
            cache_projection.union(predicate_projection);
        }
        CacheDetails {
            projection_column_ids: get_root_column_ids(schema_descr, &self.projection),
            cache_column_ids: get_root_column_ids(schema_descr, &cache_projection),
            predicate_column_ids: predicate_projection
                .as_ref()
                .map(|projection| get_root_column_ids(schema_descr, projection))
                .unwrap_or_default(),
            cache_projection,
        }
    }

    fn fallback_config(
        &self,
        row_group_idx: usize,
        details: &CacheDetails,
        cache_batch_size: usize,
    ) -> ParquetFallbackConfig {
        ParquetFallbackConfig {
            row_group_idx,
            metadata: Arc::clone(&self.metadata),
            input: self.input.clone(),
            cache_projection: details.cache_projection.clone(),
            cache_column_ids: details.cache_column_ids.clone(),
            cache_batch_size,
            row_count: self.metadata.row_group(row_group_idx).num_rows() as usize,
        }
    }

    pub(crate) fn estimated_bytes(&self, row_group_idx: usize) -> usize {
        let details = self.cache_details();
        self.metadata
            .row_group(row_group_idx)
            .columns()
            .iter()
            .enumerate()
            .filter(|(idx, _)| details.cache_projection.leaf_included(*idx))
            .map(|(_, column)| column.uncompressed_size() as usize)
            .sum()
    }

    pub(crate) fn prefetch_context(
        &self,
        row_group_idx: usize,
        snapshots: Arc<RowGroupSnapshots>,
    ) -> LiquidRowGroupPrefetchContext {
        let details = self.cache_details();
        let cached_row_group = self.cached_file.create_row_group_with_snapshots(
            row_group_idx as u64,
            details.predicate_column_ids.clone(),
            snapshots,
        );
        let fallback = ParquetFallback::new(self.fallback_config(
            row_group_idx,
            &details,
            cached_row_group.batch_size(),
        ));
        LiquidRowGroupPrefetchContext {
            cached_row_group,
            cache_column_ids: details.cache_column_ids,
            predicate_column_ids: details.predicate_column_ids,
            projection_column_ids: details.projection_column_ids,
            row_filter: self.row_filter.clone(),
            fallback,
        }
    }

    pub(crate) fn plan(
        &self,
        row_group_idx: usize,
        selection: Option<RowSelection>,
        snapshots: Arc<RowGroupSnapshots>,
    ) -> Option<LiquidRowGroupMorsel> {
        let metadata = self.metadata.row_group(row_group_idx);
        let details = self.cache_details();

        let selection = selection
            .unwrap_or_else(|| vec![RowSelector::select(metadata.num_rows() as usize)].into());
        if selection.row_count() == 0 {
            return None;
        }

        let schema_descr = self.metadata.file_metadata().schema_descr();
        let projection_columns = get_root_column_ids(schema_descr, &self.projection);
        let cached_row_group = self.cached_file.create_row_group_with_snapshots(
            row_group_idx as u64,
            details.predicate_column_ids.clone(),
            snapshots,
        );
        let cache_batch_size = cached_row_group.batch_size();

        Some(LiquidRowGroupMorsel {
            config: LiquidCacheReaderConfig {
                batch_size: self.batch_size,
                selection,
                row_filter: self.row_filter.clone(),
                cached_row_group,
                projection_columns,
                schema: Arc::clone(&self.stream_schema),
                parquet_fallback: self.fallback_config(row_group_idx, &details, cache_batch_size),
            },
            output_schema: Arc::clone(&self.output_schema),
            projector: Arc::clone(&self.projector),
            replace_schema: self.replace_schema,
            span: self.span.clone(),
        })
    }
}

struct CacheDetails {
    cache_projection: ProjectionMask,
    cache_column_ids: Vec<usize>,
    predicate_column_ids: Vec<usize>,
    projection_column_ids: Vec<usize>,
}

pub(crate) struct LiquidRowGroupPrefetchContext {
    pub(crate) cached_row_group: CachedRowGroupRef,
    pub(crate) cache_column_ids: Vec<usize>,
    pub(crate) predicate_column_ids: Vec<usize>,
    pub(crate) projection_column_ids: Vec<usize>,
    pub(crate) row_filter: Option<LiquidRowFilter>,
    pub(crate) fallback: ParquetFallback,
}

pub(crate) fn build_projection_schema(
    file_schema: &SchemaRef,
    projection_column_ids: &[usize],
) -> SchemaRef {
    let fields = projection_column_ids
        .iter()
        .filter_map(|column_id| file_schema.fields().get(*column_id))
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

pub(crate) struct LiquidRowGroupMorsel {
    config: LiquidCacheReaderConfig,
    output_schema: SchemaRef,
    projector: Arc<Projector>,
    replace_schema: bool,
    span: Option<Arc<fastrace::Span>>,
}

impl fmt::Debug for LiquidRowGroupMorsel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiquidRowGroupMorsel")
            .finish_non_exhaustive()
    }
}

impl Morsel for LiquidRowGroupMorsel {
    fn into_stream(self: Box<Self>) -> BoxStream<'static, datafusion::error::Result<RecordBatch>> {
        let Self {
            config,
            output_schema,
            projector,
            replace_schema,
            span,
        } = *self;
        let mut reader = LiquidCacheReader::new(config);
        let stream = futures::stream::poll_fn(move |cx| {
            let _guard = span.as_ref().map(|span| span.set_local_parent());
            Pin::new(&mut reader).poll_next(cx)
        });

        stream
            .map(|batch| batch.map_err(|error| DataFusionError::External(Box::new(error))))
            .map(move |batch| {
                batch.and_then(|batch| {
                    let batch = projector.project_batch(&batch)?;
                    if replace_schema {
                        let (_schema, arrays, num_rows) = batch.into_parts();
                        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
                        RecordBatch::try_new_with_options(
                            Arc::clone(&output_schema),
                            arrays,
                            &options,
                        )
                        .map_err(Into::into)
                    } else {
                        Ok(batch)
                    }
                })
            })
            .boxed()
    }
}
