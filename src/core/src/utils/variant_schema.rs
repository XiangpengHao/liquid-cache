//! Helpers for building shredding schemas for variant columns.

use std::{collections::BTreeMap, sync::Arc};

use arrow_schema::{DataType, Field, FieldRef, Fields};

fn split_variant_path(path: &str) -> Vec<String> {
    path.split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

/// Logical schema builder for variant typed_value trees.
#[derive(Default, Clone)]
pub struct VariantSchema {
    root: VariantSchemaNode,
}

impl VariantSchema {
    /// Create a schema seeded with an existing typed_value field (if present).
    pub fn new(existing_typed_value: Option<&Field>) -> Self {
        let mut root = VariantSchemaNode::default();
        if let Some(field) = existing_typed_value {
            root.absorb_root(field);
        }
        Self { root }
    }

    /// Insert a typed path into the schema.
    pub fn insert_path(&mut self, path: &str, data_type: &DataType) {
        let segments = split_variant_path(path);
        if segments.is_empty() {
            return;
        }
        self.root.insert_segments(&segments, data_type);
    }

    /// The physical fields for the typed_value struct.
    pub fn typed_fields(&self) -> Vec<FieldRef> {
        self.root.build_arrow_children()
    }

    /// The logical struct type used when shredding.
    pub fn shredding_type(&self) -> Option<DataType> {
        self.root.logical_struct_type()
    }
}

#[derive(Default, Clone)]
struct VariantSchemaNode {
    children: BTreeMap<String, VariantSchemaNode>,
    leaf_type: Option<DataType>,
}

impl VariantSchemaNode {
    fn absorb_root(&mut self, typed_field: &Field) {
        if let DataType::Struct(children) = typed_field.data_type() {
            for child in children.iter() {
                self.children
                    .entry(child.name().clone())
                    .or_default()
                    .absorb_shredded_field(child.as_ref());
            }
        }
    }

    fn absorb_shredded_field(&mut self, field: &Field) {
        match field.data_type() {
            DataType::Struct(children) => {
                let Some(typed_child) = children
                    .iter()
                    .find(|child| child.name() == "typed_value")
                    .map(|child| child.as_ref())
                else {
                    return;
                };
                match typed_child.data_type() {
                    DataType::Struct(grand_children) if !grand_children.is_empty() => {
                        for grand_child in grand_children.iter() {
                            self.children
                                .entry(grand_child.name().clone())
                                .or_default()
                                .absorb_shredded_field(grand_child.as_ref());
                        }
                    }
                    other => {
                        self.leaf_type = Some(other.clone());
                    }
                }
            }
            other => {
                self.leaf_type = Some(other.clone());
            }
        }
    }

    fn insert_segments(&mut self, segments: &[String], data_type: &DataType) {
        if segments.is_empty() {
            self.leaf_type = Some(data_type.clone());
            return;
        }
        let (head, tail) = segments.split_first().unwrap();
        self.children
            .entry(head.clone())
            .or_default()
            .insert_segments(tail, data_type);
    }

    fn build_arrow_children(&self) -> Vec<FieldRef> {
        self.children
            .iter()
            .filter_map(|(name, child)| child.build_arrow_field(name))
            .collect()
    }

    fn build_arrow_field(&self, name: &str) -> Option<FieldRef> {
        let typed_value_type = if self.children.is_empty() {
            self.leaf_type.clone()?
        } else {
            let child_fields = self.build_arrow_children();
            if child_fields.is_empty() {
                return None;
            }
            DataType::Struct(Fields::from(child_fields))
        };
        let fields = Fields::from(vec![
            Arc::new(Field::new("value", DataType::BinaryView, true)),
            Arc::new(Field::new("typed_value", typed_value_type, true)),
        ]);
        Some(Arc::new(Field::new(name, DataType::Struct(fields), false)))
    }

    fn logical_struct_type(&self) -> Option<DataType> {
        if self.children.is_empty() {
            self.leaf_type.clone()
        } else {
            let child_fields: Vec<_> = self
                .children
                .iter()
                .filter_map(|(name, child)| child.logical_field(name))
                .collect();
            if child_fields.is_empty() {
                None
            } else {
                Some(DataType::Struct(Fields::from(child_fields)))
            }
        }
    }

    fn logical_field(&self, name: &str) -> Option<FieldRef> {
        self.logical_struct_type()
            .map(|data_type| Arc::new(Field::new(name, data_type, false)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shredding_type_matches_typed_fields() {
        let mut schema = VariantSchema::default();
        schema.insert_path("a", &DataType::Int64);
        schema.insert_path("b.c", &DataType::Utf8);

        let typed = schema.typed_fields();
        assert_eq!(typed.len(), 2);
        assert_eq!(typed[0].name(), "a");
        assert_eq!(typed[1].name(), "b");

        let logical = schema.shredding_type().unwrap();
        match logical {
            DataType::Struct(fields) => {
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("expected struct"),
        }
    }
}
