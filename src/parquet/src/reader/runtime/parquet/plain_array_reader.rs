use std::sync::Arc;

use arrow_schema::{DataType, Field};

use super::super::parquet_bridge::{ParquetField, ParquetFieldType};

/// Metadata describing a column produced by [`PlainArrayReader`].
#[derive(Debug)]
pub(crate) struct ArrayReaderColumn {
    pub(crate) column_idx: usize,
    pub(crate) field: Arc<Field>,
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
