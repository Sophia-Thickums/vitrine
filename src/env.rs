//! Serialized shader environment — the on-disk form of `FileEnvironment`.
//!
//! Written by `GenericEnvironment::Serialize` in `src/video_core/shader_environment.cpp` and read
//! by `FileEnvironment::Deserialize`. The byte layout is reproduced exactly; the type does the
//! work that the C++ `ifstream` does not.
//!
//! # Layout (little-endian)
//!
//! ```text
//! u64 code_size
//! u64 num_texture_types
//! u64 num_texture_pixel_formats
//! u64 num_cbuf_values
//! u64 num_cbuf_replacement_values
//! u64 num_cbuf_sizes
//! u32 local_memory_size
//! u32 texture_bound
//! u32 start_address
//! u32 cached_lowest
//! u32 cached_highest
//! u32 viewport_transform_state
//! u32 stage                     (u32 discriminant of Shader::Stage)
//! u64 code[code_size/8]
//! (u32 key, u32 value)[num_texture_types]
//! (u32 key, u32 value)[num_texture_pixel_formats]
//! (u64 key, u32 value)[num_cbuf_values]
//! (u64 key, u32 value)[num_cbuf_replacement_values]
//! (u32 key, u32 value)[num_cbuf_sizes]
//! if stage == Compute:  u32 workgroup_size[3], u32 shared_memory_size
//! else:                 u8 sph[0x50]; (+ u32 gp_passthrough_mask[8] if stage == Geometry)
//! ```
//!
//! # Two things worth knowing before changing this file
//!
//! **The map order is the C++ iteration order, which is unspecified.** `Serialize` writes
//! `std::unordered_map`s directly, so two runs of Citron can emit the same environment with the
//! entries in different orders. That means: (a) never compare a re-serialized environment to the
//! original byte-for-byte without sorting, and (b) the reader below must accept any order, which it
//! does by rebuilding maps rather than relying on position.
//!
//! **`code_size` is a byte count, and the C++ writer takes it from `CachedSizeBytes()`.** A file
//! written by a build with a different `CachedSizeBytes` accounting would have a wrong length here.
//! The parser therefore treats `code_size` as untrusted like every other length: it is bounded by
//! `MAX_CODE_BYTES` before any allocation.

use std::collections::BTreeMap;

use crate::cursor::Cursor;

/// Size of `Shader::ProgramHeader`, asserted `0x50` in `program_header.h`. Serialized raw.
pub const SPH_BYTES: usize = 0x50;

/// Bytes in a `gp_passthrough_mask` array (`std::array<u32, 8>`).
pub const GP_PASSTHROUGH_BYTES: usize = 32;

/// Ceiling on serialized shader code, matching `MAXIMUM_SIZE` in `TryFindSize` (0x100000).
///
/// A corrupt `code_size` above this is rejected rather than allocated — the C++ reader would
/// happily `resize` to whatever the file claimed.
pub const MAX_CODE_BYTES: usize = 0x100000;

/// A shader stage, as the `u32` discriminant written by `GenericEnvironment::Serialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,
}

impl Stage {
    fn from_u32(v: u32) -> Result<Self, EnvParseError> {
        // Order follows `Shader::Stage` in `src/shader_recompiler/stage.h`.
        Ok(match v {
            0 => Self::Vertex,
            1 => Self::TessellationControl,
            2 => Self::TessellationEvaluation,
            3 => Self::Geometry,
            4 => Self::Fragment,
            5 => Self::Compute,
            other => return Err(EnvParseError::BadStage { found: other }),
        })
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Vertex => 0,
            Self::TessellationControl => 1,
            Self::TessellationEvaluation => 2,
            Self::Geometry => 3,
            Self::Fragment => 4,
            Self::Compute => 5,
        }
    }

    /// True when serialization ends with compute fields instead of a program header.
    pub fn is_compute(self) -> bool {
        self == Self::Compute
    }

    /// True when a `gp_passthrough_mask` follows the program header.
    pub fn is_geometry(self) -> bool {
        self == Self::Geometry
    }
}

/// The map-valued part of an environment, sorted so that two equal environments serialize to the
/// same bytes regardless of the order the C++ side happened to iterate in.
///
/// `BTreeMap` rather than `HashMap` is the whole point: it makes serialization deterministic, so a
/// re-written cache is byte-stable and can be diffed, checksummed, and compared across machines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Maps {
    /// `u32` handle -> `Shader::TextureType`.
    pub texture_types: BTreeMap<u32, u32>,
    /// `u32` handle -> `Shader::TexturePixelFormat`.
    pub texture_pixel_formats: BTreeMap<u32, u32>,
    /// `u64` (bank << 32 | offset) -> cbuf value.
    pub cbuf_values: BTreeMap<u64, u32>,
    /// `u64` (bank << 32 | offset) -> `Shader::ReplaceConstant` (an enum; kept as its `u32`).
    pub cbuf_replacements: BTreeMap<u64, u32>,
    /// `u32` cbuf index -> size.
    pub cbuf_sizes: BTreeMap<u32, u32>,
}

impl Maps {
    fn entry_count(&self) -> usize {
        self.texture_types.len()
            + self.texture_pixel_formats.len()
            + self.cbuf_values.len()
            + self.cbuf_replacements.len()
            + self.cbuf_sizes.len()
    }
}

/// One serialized shader environment, field for field with `GenericEnvironment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    pub local_memory_size: u32,
    pub texture_bound: u32,
    pub start_address: u32,
    pub cached_lowest: u32,
    pub cached_highest: u32,
    pub viewport_transform_state: u32,
    pub stage: Stage,
    /// Raw ARM64 instructions, one `u64` each.
    pub code: Vec<u64>,
    pub maps: Maps,
    /// Compute only: `std::array<u32, 3>`.
    pub workgroup_size: Option<[u32; 3]>,
    /// Compute only.
    pub shared_memory_size: Option<u32>,
    /// Non-compute only: `Shader::ProgramHeader`, raw 0x50 bytes.
    pub sph: Option<[u8; SPH_BYTES]>,
    /// Geometry only: `std::array<u32, 8>`, raw 32 bytes.
    pub gp_passthrough_mask: Option<[u8; GP_PASSTHROUGH_BYTES]>,
}

impl Environment {
    /// Bytes this environment will occupy when written.
    pub fn size(&self) -> usize {
        let code_bytes = self.code.len() * 8;
        let maps_bytes = self.maps.entry_count() * 8;
        // Six u64 counts (48) + seven u32 scalars (28: local_memory_size, texture_bound,
        // start_address, cached_lowest, cached_highest, viewport_transform_state, stage).
        // An earlier version counted six scalars and under-reported by 4 bytes; the round-trip
        // test caught it, which is exactly why `size()` is asserted against `write()`.
        76 + code_bytes
            + maps_bytes
            + if self.stage.is_compute() {
                16
            } else {
                SPH_BYTES
                    + if self.stage.is_geometry() {
                        GP_PASSTHROUGH_BYTES
                    } else {
                        0
                    }
            }
    }

    /// Read one environment from `cursor`. Every length is validated before it is used.
    pub fn parse(cursor: &mut Cursor<'_>) -> Result<Self, EnvParseError> {
        // Six u64 counts.
        let code_size = cursor.u64().map_err(|_| short(8, cursor))? as usize;
        let n_tex_types = cursor.u64().map_err(|_| short(8, cursor))? as usize;
        let n_tex_formats = cursor.u64().map_err(|_| short(8, cursor))? as usize;
        let n_cbuf_values = cursor.u64().map_err(|_| short(8, cursor))? as usize;
        let n_cbuf_repl = cursor.u64().map_err(|_| short(8, cursor))? as usize;
        let n_cbuf_sizes = cursor.u64().map_err(|_| short(8, cursor))? as usize;

        if !code_size.is_multiple_of(8) {
            return Err(EnvParseError::BadCodeSize { found: code_size });
        }
        if code_size > MAX_CODE_BYTES {
            return Err(EnvParseError::CodeSizeTooLarge {
                found: code_size,
                limit: MAX_CODE_BYTES,
            });
        }
        // Each map entry is 8 bytes. Refuse counts that could not possibly fit in what remains,
        // before allocating anything sized by an attacker-controlled (or corrupt) integer.
        for count in [
            n_tex_types,
            n_tex_formats,
            n_cbuf_values,
            n_cbuf_repl,
            n_cbuf_sizes,
        ] {
            if count > cursor.len() / 8 {
                return Err(EnvParseError::MapTooLarge {
                    count,
                    remaining: cursor.len(),
                });
            }
        }

        let local_memory_size = cursor.u32().map_err(|_| short(4, cursor))?;
        let texture_bound = cursor.u32().map_err(|_| short(4, cursor))?;
        let start_address = cursor.u32().map_err(|_| short(4, cursor))?;
        let cached_lowest = cursor.u32().map_err(|_| short(4, cursor))?;
        let cached_highest = cursor.u32().map_err(|_| short(4, cursor))?;
        let viewport_transform_state = cursor.u32().map_err(|_| short(4, cursor))?;
        let stage = Stage::from_u32(cursor.u32().map_err(|_| short(4, cursor))?)?;

        let words = code_size / 8;
        let mut code = Vec::new();
        code.try_reserve_exact(words)
            .map_err(|_| EnvParseError::OutOfMemory { bytes: code_size })?;
        for _ in 0..words {
            code.push(cursor.u64().map_err(|_| short(8, cursor))?);
        }

        let mut maps = Maps::default();
        for _ in 0..n_tex_types {
            let k = cursor.u32().map_err(|_| short(4, cursor))?;
            let v = cursor.u32().map_err(|_| short(4, cursor))?;
            maps.texture_types.insert(k, v);
        }
        for _ in 0..n_tex_formats {
            let k = cursor.u32().map_err(|_| short(4, cursor))?;
            let v = cursor.u32().map_err(|_| short(4, cursor))?;
            maps.texture_pixel_formats.insert(k, v);
        }
        for _ in 0..n_cbuf_values {
            let k = cursor.u64().map_err(|_| short(8, cursor))?;
            let v = cursor.u32().map_err(|_| short(4, cursor))?;
            maps.cbuf_values.insert(k, v);
        }
        for _ in 0..n_cbuf_repl {
            let k = cursor.u64().map_err(|_| short(8, cursor))?;
            let v = cursor.u32().map_err(|_| short(4, cursor))?;
            maps.cbuf_replacements.insert(k, v);
        }
        for _ in 0..n_cbuf_sizes {
            let k = cursor.u32().map_err(|_| short(4, cursor))?;
            let v = cursor.u32().map_err(|_| short(4, cursor))?;
            maps.cbuf_sizes.insert(k, v);
        }

        let mut workgroup_size = None;
        let mut shared_memory_size = None;
        let mut sph = None;
        let mut gp_passthrough_mask = None;

        if stage.is_compute() {
            let x = cursor.u32().map_err(|_| short(4, cursor))?;
            let y = cursor.u32().map_err(|_| short(4, cursor))?;
            let z = cursor.u32().map_err(|_| short(4, cursor))?;
            workgroup_size = Some([x, y, z]);
            shared_memory_size = Some(cursor.u32().map_err(|_| short(4, cursor))?);
        } else {
            sph = Some(
                cursor
                    .take::<SPH_BYTES>()
                    .map_err(|_| short(SPH_BYTES, cursor))?,
            );
            if stage.is_geometry() {
                gp_passthrough_mask = Some(
                    cursor
                        .take::<GP_PASSTHROUGH_BYTES>()
                        .map_err(|_| short(GP_PASSTHROUGH_BYTES, cursor))?,
                );
            }
        }

        Ok(Self {
            local_memory_size,
            texture_bound,
            start_address,
            cached_lowest,
            cached_highest,
            viewport_transform_state,
            stage,
            code,
            maps,
            workgroup_size,
            shared_memory_size,
            sph,
            gp_passthrough_mask,
        })
    }

    /// Append this environment to `out` in Citron's on-disk order.
    pub fn write(&self, out: &mut Vec<u8>) {
        push_u64(out, (self.code.len() * 8) as u64);
        push_u64(out, self.maps.texture_types.len() as u64);
        push_u64(out, self.maps.texture_pixel_formats.len() as u64);
        push_u64(out, self.maps.cbuf_values.len() as u64);
        push_u64(out, self.maps.cbuf_replacements.len() as u64);
        push_u64(out, self.maps.cbuf_sizes.len() as u64);
        push_u32(out, self.local_memory_size);
        push_u32(out, self.texture_bound);
        push_u32(out, self.start_address);
        push_u32(out, self.cached_lowest);
        push_u32(out, self.cached_highest);
        push_u32(out, self.viewport_transform_state);
        push_u32(out, self.stage.as_u32());
        for word in &self.code {
            push_u64(out, *word);
        }
        for (k, v) in &self.maps.texture_types {
            push_u32(out, *k);
            push_u32(out, *v);
        }
        for (k, v) in &self.maps.texture_pixel_formats {
            push_u32(out, *k);
            push_u32(out, *v);
        }
        for (k, v) in &self.maps.cbuf_values {
            push_u64(out, *k);
            push_u32(out, *v);
        }
        for (k, v) in &self.maps.cbuf_replacements {
            push_u64(out, *k);
            push_u32(out, *v);
        }
        for (k, v) in &self.maps.cbuf_sizes {
            push_u32(out, *k);
            push_u32(out, *v);
        }
        if self.stage.is_compute() {
            let wg = self.workgroup_size.unwrap_or([0; 3]);
            push_u32(out, wg[0]);
            push_u32(out, wg[1]);
            push_u32(out, wg[2]);
            push_u32(out, self.shared_memory_size.unwrap_or(0));
        } else {
            let sph = self.sph.unwrap_or([0; SPH_BYTES]);
            out.extend_from_slice(&sph);
            if self.stage.is_geometry() {
                let gp = self
                    .gp_passthrough_mask
                    .unwrap_or([0; GP_PASSTHROUGH_BYTES]);
                out.extend_from_slice(&gp);
            }
        }
    }
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn short(needed: usize, cursor: &Cursor<'_>) -> EnvParseError {
    EnvParseError::Truncated {
        needed,
        remaining: cursor.len(),
    }
}

use std::fmt;

/// Why one environment could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvParseError {
    /// Fewer bytes than a field needs were available.
    Truncated { needed: usize, remaining: usize },
    /// `code_size` was not a whole number of 8-byte instructions.
    BadCodeSize { found: usize },
    /// `code_size` exceeded [`super::MAX_CODE_BYTES`].
    CodeSizeTooLarge { found: usize, limit: usize },
    /// A map count exceeded the bytes remaining, so it cannot be honest.
    MapTooLarge { count: usize, remaining: usize },
    /// Allocation of the declared code size failed.
    OutOfMemory { bytes: usize },
    /// The stage discriminant is not one `Shader::Stage` defines.
    BadStage { found: u32 },
}

impl fmt::Display for EnvParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, remaining } => {
                write!(f, "need {needed} bytes, {remaining} remain")
            }
            Self::BadCodeSize { found } => {
                write!(f, "code_size {found} is not a multiple of 8")
            }
            Self::CodeSizeTooLarge { found, limit } => {
                write!(f, "code_size {found} exceeds the {limit}-byte limit")
            }
            Self::MapTooLarge { count, remaining } => {
                write!(f, "map count {count} cannot fit in {remaining} bytes")
            }
            Self::OutOfMemory { bytes } => write!(f, "could not reserve {bytes} bytes for code"),
            Self::BadStage { found } => write!(f, "unknown shader stage {found}"),
        }
    }
}

impl std::error::Error for EnvParseError {}
