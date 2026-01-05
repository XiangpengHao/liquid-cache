use std::sync::Arc;

use arrow::buffer::{Buffer, OffsetBuffer};

const FINGERPRINT_BUCKETS: u8 = 32;
const FINGERPRINT_MASK: u8 = FINGERPRINT_BUCKETS - 1;

// 32-bit bucketed fingerprint for a string's byte set.
#[derive(Clone, Copy, Debug)]
pub(super) struct StringFingerprint(u32);

impl StringFingerprint {
    // Construct directly from a precomputed 32-bit mask.
    pub(super) fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    // Map each byte into a bucket and set its bit (round-robin over 32 buckets).
    pub(super) fn from_bytes(bytes: &[u8]) -> Self {
        let mut bits = 0u32;
        for &byte in bytes {
            let bucket = (byte & FINGERPRINT_MASK) as u32;
            bits |= 1u32 << bucket;
        }
        Self(bits)
    }

    pub(super) fn bits(self) -> u32 {
        self.0
    }

    // Returns false only when a substring cannot be present.
    pub(super) fn might_contain(self, needle: Self) -> bool {
        (self.0 & needle.0) == needle.0
    }
}

pub(super) fn build_fingerprints(values: &Buffer, offsets: &OffsetBuffer<i32>) -> Arc<[u32]> {
    let offsets = offsets.as_ref();
    if offsets.len() < 2 {
        return Arc::from([]);
    }

    let mut fingerprints = Vec::with_capacity(offsets.len().saturating_sub(1));
    let values = values.as_slice();

    for window in offsets.windows(2) {
        let start = window[0] as usize;
        let end = window[1] as usize;
        debug_assert!(start <= end);
        debug_assert!(end <= values.len());
        let bytes = &values[start..end];
        fingerprints.push(StringFingerprint::from_bytes(bytes).0);
    }

    Arc::from(fingerprints.into_boxed_slice())
}

pub(super) fn substring_pattern_bytes(pattern: &[u8]) -> Option<&[u8]> {
    if pattern.len() < 2 {
        return None;
    }
    if pattern[0] != b'%' || pattern[pattern.len() - 1] != b'%' {
        return None;
    }
    let inner = &pattern[1..pattern.len() - 1];
    if inner.is_empty() {
        return None;
    }
    if inner.iter().any(|b| *b == b'%' || *b == b'_') {
        return None;
    }
    Some(inner)
}
