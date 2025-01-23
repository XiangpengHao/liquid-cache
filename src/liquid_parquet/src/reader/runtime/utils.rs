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

#[cfg(test)]
mod tests {
    use super::*;

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
