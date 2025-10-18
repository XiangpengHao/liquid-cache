use std::sync::Arc;

use arrow_schema::{DataType, Field};
use parquet::{
    arrow::{
        array_reader::{ArrayReader, StructArrayReader},
        arrow_reader::{RowGroups, metrics::ArrowReaderMetrics},
    },
    errors::ParquetError,
};

use super::super::parquet_bridge::{ParquetField, ParquetFieldType};
use crate::reader::runtime::parquet_bridge::StructArrayReaderBridge;

/// Metadata describing a column produced by [`PlainArrayReader`].
#[derive(Debug)]
pub(crate) struct ArrayReaderColumn {
    pub(crate) column_idx: usize,
    pub(crate) field: Arc<Field>,
}

/// Wrapper around an un-instrumented [`ArrayReader`] with column metadata.
pub(crate) struct PlainArrayReader {
    pub(crate) reader: Box<dyn ArrayReader>,
    pub(crate) columns: Vec<ArrayReaderColumn>,
}

pub(crate) fn get_column_ids(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
) -> Vec<usize> {
    let Some(field) = field else {
        return vec![];
    };

    match &field.field_type {
        ParquetFieldType::Group { children } => match &field.arrow_type {
            DataType::Struct(_) => children
                .iter()
                .filter_map(|child| match child.field_type {
                    ParquetFieldType::Primitive { col_idx, .. } => {
                        projection.leaf_included(col_idx).then_some(col_idx)
                    }
                    _ => None,
                })
                .collect(),
            _ => unreachable!("Root arrow type must be Struct"),
        },
        ParquetFieldType::Primitive { .. } => vec![],
    }
}

pub(crate) fn build_plain_array_reader(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
    row_groups: &dyn RowGroups,
) -> Result<PlainArrayReader, ParquetError> {
    let metrics = ArrowReaderMetrics::disabled();
    let builder = parquet::arrow::array_reader::ArrayReaderBuilder::new(row_groups, &metrics);

    let reader = builder.build_array_reader(
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute(field)
        },
        projection,
    )?;

    let column_ids = get_column_ids(field, projection);
    if column_ids.is_empty() {
        return Ok(PlainArrayReader {
            reader,
            columns: Vec::new(),
        });
    }

    if reader
        .as_any()
        .downcast_ref::<StructArrayReader>()
        .is_none()
    {
        return Err(ParquetError::General(
            "Expected StructArrayReader when constructing plain array reader".to_string(),
        ));
    }

    let raw = Box::into_raw(reader);
    let struct_reader = unsafe { &mut *(raw as *mut StructArrayReader) };
    let bridged_reader = StructArrayReaderBridge::from_parquet(struct_reader);

    let struct_fields = match &bridged_reader.data_type {
        DataType::Struct(fields) => fields,
        _ => {
            return Err(ParquetError::General(
                "Expected struct data type in plain array reader".to_string(),
            ));
        }
    };

    if struct_fields.len() != column_ids.len() {
        return Err(ParquetError::General(format!(
            "column count mismatch between projection ({}) and struct reader children ({})",
            column_ids.len(),
            struct_fields.len()
        )));
    }

    let columns = column_ids
        .iter()
        .zip(struct_fields.iter())
        .map(|(column_idx, field)| ArrayReaderColumn {
            column_idx: *column_idx,
            field: field.clone(),
        })
        .collect();

    let reader: Box<dyn ArrayReader> = unsafe { Box::from_raw(raw as *mut StructArrayReader) };

    Ok(PlainArrayReader { reader, columns })
}
