use std::collections::HashMap;

/// Represents the kind of cache entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKind {
    MemoryArrow,
    MemorySqueezedLiquid,
    DiskArrow,
    Unknown(String),
}

impl CacheKind {
    pub fn from_str(s: &str) -> Self {
        match s {
            "MemoryArrow" => CacheKind::MemoryArrow,
            "MemorySqueezedLiquid" => CacheKind::MemorySqueezedLiquid,
            "DiskArrow" => CacheKind::DiskArrow,
            _ => CacheKind::Unknown(s.to_string()),
        }
    }

    /// Get a short display name for UI
    pub fn display_name(&self) -> &str {
        match self {
            CacheKind::MemoryArrow => "Memory Arrow",
            CacheKind::MemorySqueezedLiquid => "Memory Squeezed",
            CacheKind::DiskArrow => "Disk Arrow",
            CacheKind::Unknown(s) => s.as_str(),
        }
    }
}

/// Represents a single trace event
#[derive(Debug, Clone)]
pub enum TraceEvent {
    InsertSuccess {
        entry: u64,
        kind: CacheKind,
    },
    InsertFailed {
        entry: u64,
        kind: CacheKind,
    },
    SqueezeBegin {
        victims: Vec<u64>,
    },
    SqueezeVictim {
        entry: u64,
    },
    IoWrite {
        entry: u64,
        kind: CacheKind,
        bytes: u64,
    },
    IoReadArrow {
        entry: u64,
        bytes: u64,
    },
    Hydrate {
        entry: u64,
        cached: CacheKind,
        new: CacheKind,
    },
    Read {
        entry: u64,
        selection: bool,
        expr: Option<String>,
        cached: CacheKind,
    },
    ReadSqueezedDate {
        entry: u64,
        expression: String,
    },
    Unknown {
        raw: String,
    },
}

impl TraceEvent {
    pub fn description(&self) -> String {
        match self {
            TraceEvent::InsertSuccess { entry, kind } => {
                format!("Insert entry {} as {}", entry, kind.display_name())
            }
            TraceEvent::InsertFailed { entry, kind } => {
                format!(
                    "Failed to insert entry {} as {}",
                    entry,
                    kind.display_name()
                )
            }
            TraceEvent::SqueezeBegin { victims } => {
                format!("Begin squeeze (victims: {:?})", victims)
            }
            TraceEvent::SqueezeVictim { entry } => {
                format!("Squeeze victim {}", entry)
            }
            TraceEvent::IoWrite { entry, kind, bytes } => {
                format!(
                    "Write entry {} as {} ({} bytes)",
                    entry,
                    kind.display_name(),
                    bytes
                )
            }
            TraceEvent::IoReadArrow { entry, bytes } => {
                format!("Read arrow entry {} ({} bytes)", entry, bytes)
            }
            TraceEvent::Hydrate { entry, cached, new } => {
                format!(
                    "Hydrate entry {} from {} to {}",
                    entry,
                    cached.display_name(),
                    new.display_name()
                )
            }
            TraceEvent::Read {
                entry,
                cached,
                expr,
                selection,
                ..
            } => {
                if let Some(expression) = expr {
                    format!(
                        "Read entry {} [{}] ({}) filtered={}",
                        entry,
                        expression,
                        cached.display_name(),
                        selection
                    )
                } else {
                    format!(
                        "Read entry {} ({}) filtered={}",
                        entry,
                        cached.display_name(),
                        selection
                    )
                }
            }
            TraceEvent::ReadSqueezedDate { entry, expression } => {
                format!("Read squeezed entry {} [{}]", entry, expression)
            }
            TraceEvent::Unknown { raw } => raw.clone(),
        }
    }

    pub fn event_type(&self) -> &str {
        match self {
            TraceEvent::InsertSuccess { .. } => "insert_success",
            TraceEvent::InsertFailed { .. } => "insert_failed",
            TraceEvent::SqueezeBegin { .. } => "squeeze_begin",
            TraceEvent::SqueezeVictim { .. } => "squeeze_victim",
            TraceEvent::IoWrite { .. } => "io_write",
            TraceEvent::IoReadArrow { .. } => "io_read_arrow",
            TraceEvent::Hydrate { .. } => "hydrate",
            TraceEvent::Read { .. } => "read",
            TraceEvent::ReadSqueezedDate { .. } => "read_squeezed_date",
            TraceEvent::Unknown { .. } => "unknown",
        }
    }
}

/// Parse a logfmt line into a HashMap
fn parse_logfmt(line: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_key = String::new();
    let mut current_value = String::new();
    let mut in_key = true;
    let mut in_brackets = false;

    for ch in line.chars() {
        match ch {
            '=' if !in_brackets => {
                in_key = false;
            }
            ' ' if !in_brackets => {
                if !current_key.is_empty() {
                    map.insert(current_key.clone(), current_value.clone());
                    current_key.clear();
                    current_value.clear();
                    in_key = true;
                }
            }
            '[' => {
                in_brackets = true;
                current_value.push(ch);
            }
            ']' => {
                in_brackets = false;
                current_value.push(ch);
            }
            _ => {
                if in_key {
                    current_key.push(ch);
                } else {
                    current_value.push(ch);
                }
            }
        }
    }

    // Don't forget the last key-value pair
    if !current_key.is_empty() {
        map.insert(current_key, current_value);
    }

    map
}

/// Parse a victims string like "[0,1,2]" into a Vec<u64>
fn parse_victims(s: &str) -> Vec<u64> {
    s.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|v| v.trim().parse::<u64>().ok())
        .collect()
}

/// Parse a single logfmt line into a TraceEvent
fn parse_event_line(line: &str) -> TraceEvent {
    let fields = parse_logfmt(line);

    match fields.get("event").map(|s| s.as_str()) {
        Some("insert_success") => TraceEvent::InsertSuccess {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            kind: fields
                .get("kind")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
        },
        Some("insert_failed") => TraceEvent::InsertFailed {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            kind: fields
                .get("kind")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
        },
        Some("squeeze_begin") => TraceEvent::SqueezeBegin {
            victims: fields
                .get("victims")
                .map(|s| parse_victims(s))
                .unwrap_or_default(),
        },
        Some("squeeze_victim") => TraceEvent::SqueezeVictim {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        },
        Some("io_write") => TraceEvent::IoWrite {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            kind: fields
                .get("kind")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
            bytes: fields
                .get("bytes")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        },
        Some("io_read_arrow") => TraceEvent::IoReadArrow {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            bytes: fields
                .get("bytes")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        },
        Some("hydrate") => TraceEvent::Hydrate {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            cached: fields
                .get("cached")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
            new: fields
                .get("new")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
        },
        Some("read") => TraceEvent::Read {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            selection: fields
                .get("selection")
                .map(|s| s == "true")
                .unwrap_or(false),
            expr: fields.get("expr").and_then(|s| {
                // Parse expr field - if it's "true"/"false", treat as old format (no expression)
                // Otherwise, treat as the actual expression string
                if s == "true" || s == "false" {
                    None
                } else {
                    Some(s.to_string())
                }
            }),
            cached: fields
                .get("cached")
                .map(|s| CacheKind::from_str(s))
                .unwrap_or(CacheKind::Unknown("".to_string())),
        },
        Some("read_squeezed_date") => TraceEvent::ReadSqueezedDate {
            entry: fields
                .get("entry")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            expression: fields
                .get("expression")
                .map(|s| s.to_string())
                .unwrap_or_default(),
        },
        _ => TraceEvent::Unknown {
            raw: line.to_string(),
        },
    }
}

/// Parse a complete trace from a string
pub fn parse_trace(input: &str) -> Vec<TraceEvent> {
    input
        .lines()
        .map(|line| line.trim())
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with("EventTrace:")
                && !line.starts_with("[")
                && !line.starts_with("]")
                && !line.starts_with("---")
                && !line.starts_with("source:")
                && !line.starts_with("expression:")
        })
        .map(parse_event_line)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_logfmt() {
        let line = "event=insert_success entry=0 kind=MemoryArrow";
        let fields = parse_logfmt(line);
        assert_eq!(fields.get("event"), Some(&"insert_success".to_string()));
        assert_eq!(fields.get("entry"), Some(&"0".to_string()));
        assert_eq!(fields.get("kind"), Some(&"MemoryArrow".to_string()));
    }

    #[test]
    fn test_parse_victims() {
        let victims = "[0,1,2]";
        let parsed = parse_victims(victims);
        assert_eq!(parsed, vec![0, 1, 2]);
    }
}
