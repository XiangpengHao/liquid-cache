use std::collections::VecDeque;

use arrow_schema::DataType;
use parquet::{arrow::ProjectionMask, arrow::arrow_reader::RowSelector};

use crate::reader::runtime::parquet_bridge::{ParquetField, ParquetFieldType};

pub(crate) fn get_column_ids(
    field: Option<&ParquetField>,
    projection: &ProjectionMask,
) -> Vec<usize> {
    let Some(field) = field else {
        return vec![];
    };

    match &field.field_type {
        ParquetFieldType::Group { children } => match &field.arrow_type {
            DataType::Struct(_) => children
                .iter()
                .enumerate()
                .filter_map(|(idx, child)| {
                    field_has_projected_leaf(child, projection).then_some(idx)
                })
                .collect(),
            _ => Vec::new(),
        },
        ParquetFieldType::Primitive { .. } => Vec::new(),
    }
}

fn field_has_projected_leaf(field: &ParquetField, projection: &ProjectionMask) -> bool {
    match &field.field_type {
        ParquetFieldType::Primitive { col_idx, .. } => projection.leaf_included(*col_idx),
        ParquetFieldType::Group { children } => children
            .iter()
            .any(|child| field_has_projected_leaf(child, projection)),
    }
}

/// Take the next batch from the selection queue.
/// The returning selection will have exactly the batch size, or less if the selection is exhausted.
pub(super) fn take_next_batch(
    selection: &mut VecDeque<RowSelector>,
    batch_size: usize,
) -> Option<Vec<RowSelector>> {
    let mut current_selected = 0;
    let mut rt = Vec::new();
    while let Some(mut front) = selection.pop_front() {
        if front.row_count + current_selected > batch_size {
            let to_select = batch_size - current_selected;
            if to_select > 0 {
                let mut sub_front = front;
                sub_front.row_count = to_select;
                rt.push(sub_front);
            }
            let remaining = front.row_count - to_select;
            front.row_count = remaining;
            selection.push_front(front);
            current_selected += to_select;
            break;
        } else {
            rt.push(front);
            current_selected += front.row_count;
        }
    }
    if current_selected == 0 {
        return None;
    }
    Some(rt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_next_batch() {
        {
            let mut queue = VecDeque::new();
            let selection = take_next_batch(&mut queue, 8);
            assert!(selection.is_none());
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(8)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection, vec![RowSelector::select(8)]);
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(10)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection, vec![RowSelector::select(8)]);
            assert_eq!(queue, vec![RowSelector::select(2)]);
        }

        {
            let mut queue = VecDeque::from(vec![
                RowSelector::select(2),
                RowSelector::skip(2),
                RowSelector::select(2),
                RowSelector::skip(2),
                RowSelector::select(2),
            ]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(
                selection,
                vec![
                    RowSelector::select(2),
                    RowSelector::skip(2),
                    RowSelector::select(2),
                    RowSelector::skip(2),
                ]
            );
            assert_eq!(queue, vec![RowSelector::select(2)]);
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(3), RowSelector::skip(2)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(
                selection,
                vec![RowSelector::select(3), RowSelector::skip(2)]
            );
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![
                RowSelector::select(2),
                RowSelector::skip(4),
                RowSelector::select(6),
            ]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(
                selection,
                vec![
                    RowSelector::select(2),
                    RowSelector::skip(4),
                    RowSelector::select(2),
                ]
            );
            assert_eq!(queue, vec![RowSelector::select(4)]);
        }

        {
            let mut queue = VecDeque::from(vec![
                RowSelector::skip(5),
                RowSelector::select(3),
                RowSelector::skip(2),
                RowSelector::select(7),
            ]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(
                selection,
                vec![RowSelector::skip(5), RowSelector::select(3),]
            );
            assert_eq!(queue, vec![RowSelector::skip(2), RowSelector::select(7)]);
        }
    }
}
