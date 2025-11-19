use std::{collections::BTreeMap, sync::Arc};

use arrow_schema::{DataType, Field, FieldRef, Fields};

fn split_variant_path(path: &str) -> Vec<String> {
    path.split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

#[derive(Default, Clone)]
pub(crate) struct VariantSchema {
    root: VariantSchemaNode,
}

impl VariantSchema {
    pub(crate) fn new(existing_typed_value: Option<&Field>) -> Self {
        let mut root = VariantSchemaNode::default();
        if let Some(field) = existing_typed_value {
            root.absorb_root(field);
        }
        Self { root }
    }

    pub(crate) fn insert_path(&mut self, path: &str, data_type: &DataType) {
        let segments = split_variant_path(path);
        if segments.is_empty() {
            return;
        }
        self.root.insert_segments(&segments, data_type);
    }

    /// The Arrow schema of the shredded variant.
    pub(crate) fn typed_fields(&self) -> Vec<FieldRef> {
        self.root.build_arrow_children()
    }

    /// The data type we used to shred the variant column, feed to `shred_variant`
    pub(crate) fn shredding_type(&self) -> Option<DataType> {
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
    use arrow_schema::{DataType, Field, Fields};

    #[test]
    fn test_variant_schema_simple() {
        let mut schema = VariantSchema::default();
        schema.insert_path("a", &DataType::Int64);
        schema.insert_path("b", &DataType::Float64);

        // Check logical type
        let logical = schema.shredding_type().unwrap();
        let expected = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        assert_eq!(logical, expected);

        // Check physical fields
        let fields = schema.typed_fields();
        assert_eq!(fields.len(), 2);

        let f_a = &fields[0];
        assert_eq!(f_a.name(), "a");
        if let DataType::Struct(children) = f_a.data_type() {
            assert_eq!(children.len(), 2);
            assert_eq!(children[0].name(), "value");
            assert_eq!(children[0].data_type(), &DataType::BinaryView);
            assert_eq!(children[1].name(), "typed_value");
            assert_eq!(children[1].data_type(), &DataType::Int64);
        } else {
            panic!("Expected struct for field a");
        }
    }

    #[test]
    fn test_variant_schema_nested() {
        let mut schema = VariantSchema::default();
        schema.insert_path("a.b", &DataType::Utf8);

        // Check logical type
        let logical = schema.shredding_type().unwrap();
        let expected = DataType::Struct(Fields::from(vec![Field::new(
            "a",
            DataType::Struct(Fields::from(vec![Field::new("b", DataType::Utf8, false)])),
            false,
        )]));
        assert_eq!(logical, expected);

        // Check physical structure
        let fields = schema.typed_fields();
        assert_eq!(fields.len(), 1);
        let f_a = &fields[0];
        assert_eq!(f_a.name(), "a");

        // Verify nested structure of 'typed_value'
        if let DataType::Struct(children) = f_a.data_type() {
            let typed_val = &children[1];
            assert_eq!(typed_val.name(), "typed_value");
            if let DataType::Struct(grand_children) = typed_val.data_type() {
                assert_eq!(grand_children.len(), 1);
                let f_b = &grand_children[0];
                assert_eq!(f_b.name(), "b");

                if let DataType::Struct(b_children) = f_b.data_type() {
                    assert_eq!(b_children[1].data_type(), &DataType::Utf8);
                } else {
                    panic!("Expected struct for field b");
                }
            } else {
                panic!("Expected struct for a.typed_value");
            }
        } else {
            panic!("Expected struct for field a");
        }
    }

    #[test]
    fn test_variant_schema_absorb() {
        // Create a schema manually by inserting paths
        let mut original = VariantSchema::default();
        original.insert_path("x.y", &DataType::Int32);
        original.insert_path("z", &DataType::Boolean);

        let physical_fields = original.typed_fields();
        let variant_col = Field::new(
            "variant",
            DataType::Struct(Fields::from(physical_fields)),
            false,
        );

        // Absorb it into a new schema
        let restored = VariantSchema::new(Some(&variant_col));

        // Check if logical types match
        assert_eq!(restored.shredding_type(), original.shredding_type());

        // Verify the structure is preserved
        let restored_fields = restored.typed_fields();
        assert_eq!(restored_fields.len(), 2);

        // x.y check
        let f_x = restored_fields.iter().find(|f| f.name() == "x").unwrap();
        if let DataType::Struct(c) = f_x.data_type() {
            // typed_value -> Struct(y) -> typed_value -> Int32
            let tv = &c[1];
            if let DataType::Struct(gc) = tv.data_type() {
                let f_y = &gc[0];
                assert_eq!(f_y.name(), "y");
                if let DataType::Struct(yc) = f_y.data_type() {
                    assert_eq!(yc[1].data_type(), &DataType::Int32);
                }
            }
        }

        // z check
        let f_z = restored_fields.iter().find(|f| f.name() == "z").unwrap();
        if let DataType::Struct(c) = f_z.data_type() {
            assert_eq!(c[1].data_type(), &DataType::Boolean);
        }
    }
}
