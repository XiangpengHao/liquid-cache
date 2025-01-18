#![allow(unused)]

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{RowFilter, RowSelection, RowSelector};
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

pub struct ParquetRecordBatchReaderInner {
    batch_size: usize,
    array_reader: Box<dyn ArrayReader>,
    schema: SchemaRef,
    selection: Option<VecDeque<RowSelector>>,
}

impl ParquetRecordBatchReaderInner {
    pub fn new_parquet(
        batch_size: usize,
        array_reader: Box<dyn ArrayReader>,
        selection: Option<RowSelection>,
    ) -> parquet::arrow::arrow_reader::ParquetRecordBatchReader {
        let schema = match array_reader.get_data_type() {
            DataType::Struct(ref fields) => Schema::new(fields.clone()),
            _ => unreachable!("Struct array reader's data type is not struct!"),
        };

        let v = Self {
            batch_size,
            array_reader,
            schema: Arc::new(schema),
            selection: selection.map(|s| trim_row_selection(s).into()),
        };
        unsafe { std::mem::transmute(v) }
    }
}

fn trim_row_selection(selection: RowSelection) -> RowSelection {
    let mut selection: Vec<RowSelector> = selection.into();
    while selection.last().map(|x| x.skip).unwrap_or(false) {
        selection.pop();
    }
    RowSelection::from(selection)
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

struct ProjectionMask {
    mask: Option<Vec<bool>>,
}

impl ProjectionMask {
    /// Union two projection masks
    ///
    /// Example:
    /// ```text
    /// mask1 = [true, false, true]
    /// mask2 = [false, true, true]
    /// union(mask1, mask2) = [true, true, true]
    /// ```
    pub fn union(&mut self, other: &Self) {
        match (self.mask.as_ref(), other.mask.as_ref()) {
            (None, _) | (_, None) => self.mask = None,
            (Some(a), Some(b)) => {
                debug_assert_eq!(a.len(), b.len());
                let mask = a.iter().zip(b.iter()).map(|(&a, &b)| a || b).collect();
                self.mask = Some(mask);
            }
        }
    }

    /// Intersect two projection masks
    ///
    /// Example:
    /// ```text
    /// mask1 = [true, false, true]
    /// mask2 = [false, true, true]
    /// intersect(mask1, mask2) = [false, false, true]
    /// ```
    pub fn intersect(&mut self, other: &Self) {
        match (self.mask.as_ref(), other.mask.as_ref()) {
            (None, _) => self.mask = other.mask.clone(),
            (_, None) => {}
            (Some(a), Some(b)) => {
                debug_assert_eq!(a.len(), b.len());
                let mask = a.iter().zip(b.iter()).map(|(&a, &b)| a && b).collect();
                self.mask = Some(mask);
            }
        }
    }
}

pub(super) fn intersect_projection_mask(
    projection: &mut parquet::arrow::ProjectionMask,
    other: &parquet::arrow::ProjectionMask,
) {
    let project_inner: &mut ProjectionMask = unsafe { std::mem::transmute(projection) };
    let other_inner: &ProjectionMask = unsafe { std::mem::transmute(other) };
    project_inner.intersect(other_inner);
}

pub(super) fn union_projection_mask(
    projection: &mut parquet::arrow::ProjectionMask,
    other: &parquet::arrow::ProjectionMask,
) {
    let project_inner: &mut ProjectionMask = unsafe { std::mem::transmute(projection) };
    let other_inner: &ProjectionMask = unsafe { std::mem::transmute(other) };
    project_inner.union(other_inner);
}

pub struct ArrowReaderBuilderBridge {
    pub(crate) input: Box<dyn ArrayReader>,

    pub(crate) metadata: Arc<ParquetMetaData>,

    pub(crate) schema: SchemaRef,

    pub(crate) fields: Option<Arc<ParquetField>>,

    pub(crate) batch_size: usize,

    pub(crate) row_groups: Option<Vec<usize>>,

    projection: ProjectionMask,

    pub(crate) filter: Option<RowFilter>,

    pub(crate) selection: Option<RowSelection>,

    pub(crate) limit: Option<usize>,

    pub(crate) offset: Option<usize>,
}
