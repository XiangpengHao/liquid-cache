#![allow(unused)]

use std::sync::Arc;

use arrow_schema::{DataType, Field, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{ArrowReaderBuilder, RowFilter, RowSelection, RowSelector};
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::TypePtr;

/// Representation of a parquet schema element, in terms of arrow schema elements
#[derive(Debug, Clone)]
pub struct ParquetField {
    /// The level which represents an insertion into the current list
    /// i.e. guaranteed to be > 0 for an element of list type
    pub rep_level: i16,
    /// The level at which this field is fully defined,
    /// i.e. guaranteed to be > 0 for a nullable type or child of a
    /// nullable type
    pub def_level: i16,
    /// Whether this field is nullable
    pub nullable: bool,
    /// The arrow type of the column data
    ///
    /// Note: In certain cases the data stored in parquet may have been coerced
    /// to a different type and will require conversion on read (e.g. Date64 and Interval)
    pub arrow_type: DataType,
    /// The type of this field
    pub field_type: ParquetFieldType,
}

impl ParquetField {
    /// Converts `self` into an arrow list, with its current type as the field type
    ///
    /// This is used to convert repeated columns, into their arrow representation
    fn into_list(self, name: &str) -> Self {
        ParquetField {
            rep_level: self.rep_level,
            def_level: self.def_level,
            nullable: false,
            arrow_type: DataType::List(Arc::new(Field::new(name, self.arrow_type.clone(), false))),
            field_type: ParquetFieldType::Group {
                children: vec![self],
            },
        }
    }

    /// Returns a list of [`ParquetField`] children if this is a group type
    pub fn children(&self) -> Option<&[Self]> {
        match &self.field_type {
            ParquetFieldType::Primitive { .. } => None,
            ParquetFieldType::Group { children } => Some(children),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ParquetFieldType {
    Primitive {
        /// The index of the column in parquet
        col_idx: usize,
        /// The type of the column in parquet
        primitive_type: TypePtr,
    },
    Group {
        children: Vec<ParquetField>,
    },
}

pub(super) fn offset_row_selection(selection: RowSelection, offset: usize) -> RowSelection {
    if offset == 0 {
        return selection;
    }

    let mut selected_count = 0;
    let mut skipped_count = 0;

    let mut selectors: Vec<RowSelector> = selection.into();

    // Find the index where the selector exceeds the row count
    let find = selectors.iter().position(|selector| match selector.skip {
        true => {
            skipped_count += selector.row_count;
            false
        }
        false => {
            selected_count += selector.row_count;
            selected_count > offset
        }
    });

    let split_idx = match find {
        Some(idx) => idx,
        None => {
            selectors.clear();
            return RowSelection::from(selectors);
        }
    };

    let mut new_selectors = Vec::with_capacity(selectors.len() - split_idx + 1);
    new_selectors.push(RowSelector::skip(skipped_count + offset));
    new_selectors.push(RowSelector::select(selected_count - offset));
    new_selectors.extend_from_slice(&selectors[split_idx + 1..]);

    RowSelection::from(new_selectors)
}

pub(super) fn limit_row_selection(selection: RowSelection, mut limit: usize) -> RowSelection {
    let mut selectors: Vec<RowSelector> = selection.into();

    if limit == 0 {
        selectors.clear();
    }

    for (idx, selection) in selectors.iter_mut().enumerate() {
        if !selection.skip {
            if selection.row_count >= limit {
                selection.row_count = limit;
                selectors.truncate(idx + 1);
                break;
            } else {
                limit -= selection.row_count;
            }
        }
    }
    RowSelection::from(selectors)
}

#[derive(Debug, Clone)]
pub struct ProjectionMask {
    mask: Option<Vec<bool>>,
}

pub(super) fn get_predicate_column_id(projection: &parquet::arrow::ProjectionMask) -> Vec<usize> {
    let project_inner: &ProjectionMask = unsafe { std::mem::transmute(projection) };
    project_inner
        .mask
        .as_ref()
        .map(|m| {
            m.iter()
                .enumerate()
                .filter_map(|(pos, &x)| if x { Some(pos) } else { None })
                .collect::<Vec<usize>>()
        })
        .unwrap_or_default()
}

use parquet::arrow::async_reader::AsyncReader;
use tokio::sync::Mutex;

use crate::reader::plantime::ParquetMetadataCacheReader;

use super::{liquid_stream::ClonableAsyncFileReader, liquid_stream::LiquidStreamBuilder};

pub struct ArrowReaderBuilderBridge {
    pub(crate) input: AsyncReader<ParquetMetadataCacheReader>,

    pub(crate) metadata: Arc<ParquetMetaData>,

    pub(crate) schema: SchemaRef,

    pub(crate) fields: Option<Arc<ParquetField>>,

    pub(crate) batch_size: usize,

    pub(crate) row_groups: Option<Vec<usize>>,

    pub(crate) projection: parquet::arrow::ProjectionMask,

    pub(crate) filter: Option<RowFilter>,

    pub(crate) selection: Option<RowSelection>,

    pub(crate) limit: Option<usize>,

    pub(crate) offset: Option<usize>,
}

impl ArrowReaderBuilderBridge {
    pub(crate) unsafe fn from_parquet(
        builder: ArrowReaderBuilder<AsyncReader<ParquetMetadataCacheReader>>,
    ) -> Self {
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute(builder)
        }
    }

    pub(crate) fn into_liquid_builder(self) -> LiquidStreamBuilder {
        let input: ParquetMetadataCacheReader = unsafe { std::mem::transmute(self.input) };
        let input = ClonableAsyncFileReader(Arc::new(Mutex::new(input)));
        LiquidStreamBuilder {
            input,
            metadata: self.metadata,
            fields: self.fields,
            batch_size: self.batch_size,
            row_groups: self.row_groups,
            projection: self.projection,
            filter: None,
            selection: self.selection,
            limit: self.limit,
            offset: self.offset,
        }
    }
}

pub struct StructArrayReaderBridge {
    pub children: Vec<Box<dyn ArrayReader>>,
    pub data_type: DataType,
    pub struct_def_level: i16,
    pub struct_rep_level: i16,
    pub nullable: bool,
}

use parquet::arrow::array_reader::StructArrayReader;

impl StructArrayReaderBridge {
    pub fn from_parquet(parquet: &mut StructArrayReader) -> &mut Self {
        unsafe { std::mem::transmute(parquet) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::transmute;

    #[test]
    fn test_valid_projection_mask() {
        // Create a valid projection mask with one `true` at index 1.
        let proj = ProjectionMask {
            mask: Some(vec![false, true, false, true, true, false]),
        };
        let predicate_ids = unsafe { get_predicate_column_id(transmute(&proj)) };
        assert_eq!(predicate_ids, vec![1, 3, 4]);
    }
}
