//! Bounds-checked little-endian reader shared by the cache and environment parsers.
//!
//! Every out-of-bounds read becomes an `Err`, never a panic. This is the one primitive the whole
//! crate is built on: the C++ code reads with `ifstream`, which sets a fail bit that callers must
//! remember to check; here the type system will not let a caller forget.

/// Not enough bytes left to satisfy a read.
///
/// Carries what was needed and what remained, so the caller can name the shortfall in a log
/// instead of guessing — the C++ `ifstream` fail bit says nothing about how much was missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Short {
    /// Bytes the read required.
    pub needed: usize,
    /// Bytes that were available.
    pub remaining: usize,
}

/// A cursor over a byte slice that only ever moves forward.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Start at the beginning of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Current absolute offset.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Bytes left to read.
    pub fn len(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// True when the cursor is at the end.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `N` bytes.
    pub fn take<const N: usize>(&mut self) -> Result<[u8; N], Short> {
        let end = self.pos.checked_add(N).ok_or(Short {
            needed: N,
            remaining: self.len(),
        })?;
        let slice = self.bytes.get(self.pos..end).ok_or(Short {
            needed: N,
            remaining: self.len(),
        })?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        self.pos = end;
        Ok(out)
    }

    /// Read a little-endian `u8`.
    pub fn u8(&mut self) -> Result<u8, Short> {
        self.take::<1>().map(|b| b[0])
    }

    /// Read a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16, Short> {
        self.take::<2>().map(u16::from_le_bytes)
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32, Short> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64, Short> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    /// Read a little-endian `f32`.
    pub fn f32(&mut self) -> Result<f32, Short> {
        self.take::<4>().map(f32::from_le_bytes)
    }
}
