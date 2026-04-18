//! Helpers for working with shredded variant typed_value trees.

use arrow::array::StructArray;

/// Returns true if the provided struct (matching the physical layout of a shredded variant)
/// contains the dotted path.
pub fn typed_struct_contains_path(root: &StructArray, path: &str) -> bool {
    let segments: Vec<&str> = path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return false;
    }
    contains_segments(root, &segments)
}

fn contains_segments(current: &StructArray, segments: &[&str]) -> bool {
    if segments.is_empty() {
        return false;
    }
    let Some(field) = current.column_by_name(segments[0]) else {
        return false;
    };
    let Some(struct_field) = field.as_any().downcast_ref::<StructArray>() else {
        return false;
    };
    if segments.len() == 1 {
        return struct_field.column_by_name("typed_value").is_some();
    }
    let Some(next) = struct_field
        .column_by_name("typed_value")
        .and_then(|typed| typed.as_any().downcast_ref::<StructArray>())
    else {
        return false;
    };
    contains_segments(next, &segments[1..])
}
