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
