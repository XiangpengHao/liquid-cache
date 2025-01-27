use std::collections::VecDeque;

use arrow::{array::BooleanBufferBuilder, buffer::BooleanBuffer};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

/// Consolidate the selection to the batch granularity
///
/// If any row of the batch is selected, the entire batch will be selected.
/// The last batch may be partial to make sure the total row count is the same.
///
/// Example (batch size 10):
/// Input: [select(15), skip(10), select(8)]
/// Output: [select(33)]
#[allow(unused)]
fn consolidate_selection_to_batch_granularity(
    selection: &RowSelection,
    batch_size: usize,
) -> RowSelection {
    let total_row_count = selection.iter().map(|r| r.row_count).sum::<usize>();

    let mut batch_mask = vec![false; total_row_count.div_ceil(batch_size)];

    let mut current_row_id = 0;
    for s in selection.iter() {
        if s.skip {
            current_row_id += s.row_count;
            continue;
        }

        let start = current_row_id / batch_size * batch_size;
        let end = current_row_id + s.row_count;

        for i in (start..end).step_by(batch_size) {
            batch_mask[i / batch_size] = true;
        }
        current_row_id += s.row_count;
    }

    let mut new_selection = vec![];
    let mut current_batch_id = 0;
    for select in batch_mask.iter() {
        let batch_count = if current_batch_id + batch_size < total_row_count {
            batch_size
        } else {
            total_row_count - current_batch_id
        };
        if *select {
            new_selection.push(RowSelector::select(batch_count));
        } else {
            new_selection.push(RowSelector::skip(batch_count));
        }
        current_batch_id += batch_count;
    }

    RowSelection::from(new_selection)
}

/// Take the next selection from the selection queue, and return the selection
/// whose selected row count is to_select or less (if input selection is exhausted).
pub(super) fn take_next_selection(
    selection: &mut VecDeque<RowSelector>,
    to_select: usize,
) -> Option<RowSelection> {
    let mut current_selected = 0;
    let mut rt = Vec::new();
    while let Some(front) = selection.pop_front() {
        if front.skip {
            rt.push(front);
            continue;
        }

        if current_selected + front.row_count <= to_select {
            rt.push(front);
            current_selected += front.row_count;
        } else {
            let select = to_select - current_selected;
            let remaining = front.row_count - select;
            rt.push(RowSelector::select(select));
            selection.push_front(RowSelector::select(remaining));

            return Some(rt.into());
        }
    }
    if !rt.is_empty() {
        return Some(rt.into());
    }
    None
}

/// Take the next batch from the selection queue.
/// The returning selection will have exactly the batch size, or less if the selection is exhausted.
pub(super) fn take_next_batch(
    selection: &mut VecDeque<RowSelector>,
    batch_size: usize,
) -> Option<BooleanBuffer> {
    let mut current_selected = 0;
    let mut rt = BooleanBufferBuilder::new(batch_size);
    while let Some(mut front) = selection.pop_front() {
        let to_select = if front.row_count + current_selected > batch_size {
            batch_size - current_selected
        } else {
            front.row_count
        };
        if front.skip {
            rt.append_n(to_select, false);
        } else {
            rt.append_n(to_select, true);
        }
        current_selected += to_select;
        let remaining = front.row_count - to_select;
        if remaining > 0 {
            front.row_count = remaining;
            selection.push_front(front);
            break;
        }
    }
    Some(rt.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_next_batch() {
        let batch_size = 8;

        {
            let mut queue = VecDeque::new();
            let selection = take_next_batch(&mut queue, batch_size).unwrap();
            assert_eq!(selection.len(), batch_size);
            assert_eq!(selection.count_set_bits(), 0);
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(8)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection.len(), 8);
            assert_eq!(selection.count_set_bits(), 8);
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(10)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection.len(), 8);
            assert_eq!(selection.count_set_bits(), 8);
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
            assert_eq!(selection.len(), 8);
            assert_eq!(selection.count_set_bits(), 4);
            assert_eq!(queue, vec![RowSelector::select(2)]);
        }

        {
            let mut queue = VecDeque::from(vec![RowSelector::select(3), RowSelector::skip(2)]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection.len(), 8);
            assert_eq!(selection.count_set_bits(), 3);
            assert!(queue.is_empty());
        }

        {
            let mut queue = VecDeque::from(vec![
                RowSelector::select(2),
                RowSelector::skip(4),
                RowSelector::select(6),
            ]);
            let selection = take_next_batch(&mut queue, 8).unwrap();
            assert_eq!(selection.len(), 8);
            assert_eq!(selection.count_set_bits(), 4);
            assert_eq!(queue, vec![RowSelector::select(4)]);
        }

        {
            let mut queue = VecDeque::from(vec![
                RowSelector::skip(5),
                RowSelector::select(3),
                RowSelector::skip(2),
                RowSelector::select(7),
            ]);
            let selection = take_next_batch(&mut queue, batch_size).unwrap();
            assert_eq!(selection.len(), batch_size);
            assert_eq!(selection.count_set_bits(), 3);
            assert_eq!(queue, vec![RowSelector::skip(2), RowSelector::select(7)]);
        }
    }

    #[test]
    fn test_take_next_selection_exact_match() {
        let mut queue = VecDeque::from(vec![
            RowSelector::skip(5),
            RowSelector::select(3),
            RowSelector::skip(2),
            RowSelector::select(7),
        ]);

        // Request exactly 10 rows (5 skip + 3 select + 2 skip)
        let selection = take_next_selection(&mut queue, 3).unwrap();
        assert_eq!(
            selection,
            vec![
                RowSelector::skip(5),
                RowSelector::select(3),
                RowSelector::skip(2)
            ]
            .into()
        );

        // Check remaining queue
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].row_count, 7);
        assert!(!queue[0].skip);
    }

    #[test]
    fn test_take_next_selection_split_required() {
        let mut queue = VecDeque::from(vec![RowSelector::select(10), RowSelector::select(10)]);

        // Request 15 rows, which should split the first selector
        let selection = take_next_selection(&mut queue, 15).unwrap();

        assert_eq!(
            selection,
            vec![RowSelector::select(10), RowSelector::select(5)].into()
        );

        // Check remaining queue - should have 5 rows from split and original 10
        assert_eq!(queue.len(), 1);
        assert!(!queue[0].skip);
        assert_eq!(queue[0].row_count, 5);
    }

    #[test]
    fn test_take_next_selection_empty_queue() {
        let mut queue = VecDeque::new();

        // Should return None for empty queue
        let selection = take_next_selection(&mut queue, 10);
        assert!(selection.is_none());

        // Test with queue that becomes empty
        queue.push_back(RowSelector::select(5));
        let selection = take_next_selection(&mut queue, 10).unwrap();
        assert_eq!(selection, vec![RowSelector::select(5)].into());

        // Queue should now be empty
        let selection = take_next_selection(&mut queue, 10);
        assert!(selection.is_none());
    }

    #[test]
    fn test_selection() {
        let input = RowSelection::from(vec![]);
        let result = consolidate_selection_to_batch_granularity(&input, 10);
        let expected = RowSelection::from(vec![]);
        assert_eq!(result, expected);

        // Test when all rows are selected
        let input = RowSelection::from(vec![
            RowSelector::select(15),
            RowSelector::select(10),
            RowSelector::select(5),
        ]);
        let result = consolidate_selection_to_batch_granularity(&input, 10);
        let expected = RowSelection::from(vec![
            RowSelector::select(10),
            RowSelector::select(10),
            RowSelector::select(10),
        ]);
        assert_eq!(result, expected);

        // Test when all rows are skipped
        let input = RowSelection::from(vec![RowSelector::skip(20), RowSelector::skip(5)]);
        let result = consolidate_selection_to_batch_granularity(&input, 10);
        let expected = RowSelection::from(vec![
            RowSelector::skip(10),
            RowSelector::skip(10),
            RowSelector::skip(5),
        ]);
        assert_eq!(result, expected);

        let input = RowSelection::from(vec![
            RowSelector::select(3),
            RowSelector::skip(2),
            RowSelector::select(1),
        ]);
        let result = consolidate_selection_to_batch_granularity(&input, 1);
        let expected = RowSelection::from(vec![
            RowSelector::select(1),
            RowSelector::select(1),
            RowSelector::select(1),
            RowSelector::skip(1),
            RowSelector::skip(1),
            RowSelector::select(1),
        ]);
        assert_eq!(result, expected);

        // Test with batch size larger than total rows
        let input = RowSelection::from(vec![
            RowSelector::select(5),
            RowSelector::skip(3),
            RowSelector::select(2),
        ]);
        let result = consolidate_selection_to_batch_granularity(&input, 20);
        let expected = RowSelection::from(vec![RowSelector::select(10)]);
        assert_eq!(result, expected);

        let input = RowSelection::from(vec![
            RowSelector::select(15), // Spans two batches (10 + 5)
            RowSelector::skip(5),
            RowSelector::select(10), // Exactly one batch
        ]);
        let result = consolidate_selection_to_batch_granularity(&input, 10);
        let expected = RowSelection::from(vec![
            RowSelector::select(10),
            RowSelector::select(10),
            RowSelector::select(10),
        ]);
        assert_eq!(result, expected);

        let input = RowSelection::from(vec![
            RowSelector::select(1),
            RowSelector::skip(1),
            RowSelector::select(1),
            RowSelector::skip(1),
            RowSelector::select(1),
            RowSelector::skip(1),
        ]);
        let result = consolidate_selection_to_batch_granularity(&input, 2);
        let expected = RowSelection::from(vec![
            RowSelector::select(2),
            RowSelector::select(2),
            RowSelector::select(2),
        ]);
        assert_eq!(result, expected);
    }
}
