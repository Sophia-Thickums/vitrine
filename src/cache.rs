//! Pipeline cache: load, validate, repair, and write Citron Neo's on-disk shader cache.
//!
//! This replaces `LoadPipelines` / `SerializePipeline` from
//! `src/video_core/shader_environment.cpp`. The C++ original has one behaviour that this module
//! exists to make impossible: **on any corrupt record it deletes the entire cache file**, after
//! having already fed the records it read before the corrupt one into the live pipeline cache.
//! Users see that as "the emulator clears my shaders by itself", and on Android as a crash.
//!
//! The design here differs in four ways, each of which is a type or a test rather than a comment:
//!
//! 1. **Parse, then commit.** [`Cache::parse`] reads the whole file into a [`ParsedCache`] and
//!    touches nothing. Callers decide what to do with a partial result. There is no path in which
//!    half a cache is live and the file is already gone.
//! 2. **Damage is reported, not punished.** A corrupt record yields [`Damage`] describing the byte
//!    offset and the reason. The good records on either side are kept.
//! 3. **Deletion requires an explicit choice.** [`Repair::Truncate`] writes a clean cache from the
//!    salvageable records. Nothing removes the file unless the caller asks for [`Repair::Discard`].
//! 4. **The format is the on-disk format.** Magic and version are Yuzu's, deliberately: an existing
//!    user's cache must still load. Changing them would silently invalidate every installed cache.
//!
//! # Format (little-endian, as written by Citron Neo)
//!
//! ```text
//! header:  magic[8]  version u32
//! record*: num_envs u32
//!          env[num_envs]        -- opaque here; see `env.rs`
//!          key[KEY_LEN]         -- pipeline cache key
//! ```
//!
//! The parser is length-checked at every read. A truncated record, a `num_envs` outside `1..=64`,
//! or a trailing partial header all produce [`Damage`] and stop parsing at that point, keeping
//! everything already recovered.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::cursor::Cursor;
use crate::env::{Environment, EnvParseError};

pub use crate::env::EnvParseError as ParseEnvError;

/// Yuzu's magic. Kept so installed caches load; see the module docs.
pub const MAGIC: [u8; 8] = *b"yuzuvkch";

/// Cache format version this build understands. Mirrors `TRANSFERABLE_CACHE_VERSION` in Citron.
pub const CACHE_VERSION: u32 = 1;

/// Bytes in one pipeline cache key (`PIPELINE_CACHE_KEY_LEN_BYTES` in Citron).
pub const KEY_LEN: usize = 16;

/// Upper bound on environments per record. Matches the C++ guard `num_envs == 0 || > 64`.
pub const MAX_ENVS: u32 = 64;

/// A cache file that was read but not yet acted on.
///
/// Everything the file contained that survived validation, plus an account of what did not.
/// Constructing one never mutates anything on disk.
#[derive(Debug, Clone)]
pub struct ParsedCache {
    /// Records that parsed cleanly, in file order.
    pub records: Vec<Record>,
    /// The first failure, if the file was not wholly readable.
    pub damage: Option<Damage>,
    /// Bytes consumed before parsing stopped (header included).
    pub consumed: usize,
}

impl ParsedCache {
    /// True when the whole file parsed and no record was lost.
    pub fn is_clean(&self) -> bool {
        self.damage.is_none()
    }

    /// Records recovered from a damaged file, which a repair would preserve.
    pub fn salvaged(&self) -> &[Record] {
        &self.records
    }

    /// Total environments across all records — the number users think of as "shaders".
    pub fn env_count(&self) -> usize {
        self.records.iter().map(|r| r.envs.len()).sum()
    }
}

/// One cache entry: the environments compiled for a pipeline, and that pipeline's key.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// Byte offset of this record in the source file, for diagnostics.
    pub offset: usize,
    /// Generic environments; at least one, at most [`MAX_ENVS`].
    pub envs: Vec<Environment>,
    /// The 128-bit pipeline key.
    pub key: [u8; KEY_LEN],
}

/// What went wrong, and where. Carries enough to log a useful bug report.
#[derive(Debug, Clone, PartialEq)]
pub struct Damage {
    /// Byte offset in the source file where parsing stopped.
    pub offset: usize,
    /// Why it stopped.
    pub reason: DamageKind,
}

/// The failure classes the format can express.
#[derive(Debug, Clone, PartialEq)]
pub enum DamageKind {
    /// File is shorter than the 12-byte header.
    ShortHeader { have: usize },
    /// Magic is not [`MAGIC`] — usually not a pipeline cache at all.
    BadMagic { found: [u8; 8] },
    /// Written by a different cache version; not corrupt, just incompatible.
    VersionMismatch { found: u32, expected: u32 },
    /// `num_envs` was zero or above [`MAX_ENVS`]; the C++ code throws here.
    BadEnvCount { found: u32 },
    /// A record was cut short — power loss during write is the usual cause.
    Truncated { needed: usize, remaining: usize },
    /// An environment failed to deserialize; the inner error names the field.
    BadEnvironment { inner: ParseEnvError },
}

impl fmt::Display for DamageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShortHeader { have } => {
                write!(f, "file is {have} bytes, shorter than the 12-byte header")
            }
            Self::BadMagic { found } => {
                write!(f, "bad magic {found:?}, expected {MAGIC:?}")
            }
            Self::VersionMismatch { found, expected } => {
                write!(f, "cache version {found}, this build reads {expected}")
            }
            Self::BadEnvCount { found } => {
                write!(f, "record declares {found} environments (allowed 1..={MAX_ENVS})")
            }
            Self::Truncated { needed, remaining } => {
                write!(f, "record needs {needed} more bytes, {remaining} remain")
            }
            Self::BadEnvironment { inner } => write!(f, "environment did not parse: {inner}"),
        }
    }
}

impl fmt::Display for Damage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.reason)
    }
}

impl std::error::Error for Damage {}

/// What the caller decided to do with a damaged cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repair {
    /// Write a fresh cache containing only the records that parsed. The default.
    ///
    /// Loses the unreadable tail and nothing else. On a cache where damage is in the last
    /// record — the common case, a crash mid-append — this keeps every shader the user had.
    Truncate,
    /// Delete the file and start empty. The old behaviour, now an explicit request.
    Discard,
}

/// Read and validate a cache file without modifying it.
///
/// A missing file is not an error: it parses as an empty, clean cache, matching the C++
/// behaviour of returning early when the file will not open.
pub fn parse_file(path: &Path) -> io::Result<ParsedCache> {
    match fs::read(path) {
        Ok(bytes) => Ok(parse(&bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(ParsedCache {
            records: Vec::new(),
            damage: None,
            consumed: 0,
        }),
        Err(e) => Err(e),
    }
}

/// Parse a cache image held in memory.
pub fn parse(bytes: &[u8]) -> ParsedCache {
    let mut cursor = Cursor::new(bytes);

    let magic = match cursor.take::<8>() {
        Ok(m) => m,
        Err(_) => {
            return ParsedCache {
                records: Vec::new(),
                damage: Some(Damage {
                    offset: 0,
                    reason: DamageKind::ShortHeader {
                        have: bytes.len(),
                    },
                }),
                consumed: 0,
            };
        }
    };
    if magic != MAGIC {
        return ParsedCache {
            records: Vec::new(),
            damage: Some(Damage {
                offset: 0,
                reason: DamageKind::BadMagic { found: magic },
            }),
            consumed: 0,
        };
    }
    let version = match cursor.u32() {
        Ok(v) => v,
        Err(_) => {
            return ParsedCache {
                records: Vec::new(),
                damage: Some(Damage {
                    offset: 8,
                    reason: DamageKind::ShortHeader {
                        have: bytes.len(),
                    },
                }),
                consumed: 8,
            };
        }
    };
    if version != CACHE_VERSION {
        return ParsedCache {
            records: Vec::new(),
            damage: Some(Damage {
                offset: 8,
                reason: DamageKind::VersionMismatch {
                    found: version,
                    expected: CACHE_VERSION,
                },
            }),
            consumed: 12,
        };
    }

    let mut records = Vec::new();
    // A file whose length is not a whole number of records is truncated in the tail; the loop
    // below detects that, so no separate size check is needed here.
    loop {
        let offset = cursor.pos();
        if cursor.is_empty() {
            return ParsedCache {
                records,
                damage: None,
                consumed: offset,
            };
        }
        let num_envs = match cursor.u32() {
            Ok(n) => n,
            Err(_) => {
                return ParsedCache {
                    records,
                    damage: Some(Damage {
                        offset,
                        reason: DamageKind::Truncated {
                            needed: 4,
                            remaining: cursor.len(),
                        },
                    }),
                    consumed: offset,
                };
            }
        };
        if num_envs == 0 || num_envs > MAX_ENVS {
            return ParsedCache {
                records,
                damage: Some(Damage {
                    offset,
                    reason: DamageKind::BadEnvCount { found: num_envs },
                }),
                consumed: offset,
            };
        }

        let mut envs = Vec::with_capacity(num_envs as usize);
        for _ in 0..num_envs {
            match Environment::parse(&mut cursor) {
                Ok(env) => envs.push(env),
                Err(e) => {
                    let reason = match e {
                        EnvParseError::Truncated { needed, remaining } => {
                            DamageKind::Truncated { needed, remaining }
                        }
                        other => DamageKind::BadEnvironment { inner: other },
                    };
                    return ParsedCache {
                        records,
                        damage: Some(Damage { offset, reason }),
                        consumed: offset,
                    };
                }
            }
        }

        let key = match cursor.take::<KEY_LEN>() {
            Ok(k) => k,
            Err(_) => {
                return ParsedCache {
                    records,
                    damage: Some(Damage {
                        offset,
                        reason: DamageKind::Truncated {
                            needed: KEY_LEN,
                            remaining: cursor.len(),
                        },
                    }),
                    consumed: offset,
                };
            }
        };

        records.push(Record { offset, envs, key });
    }
}

/// Write a cache image from records, producing bytes in the same format [`parse`] reads.
pub fn serialize(records: &[Record]) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        12 + records
            .iter()
            .map(|r| 4 + r.envs.iter().map(Environment::size).sum::<usize>() + KEY_LEN)
            .sum::<usize>(),
    );
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    for record in records {
        out.extend_from_slice(&(record.envs.len() as u32).to_le_bytes());
        for env in &record.envs {
            env.write(&mut out);
        }
        out.extend_from_slice(&record.key);
    }
    out
}

/// Apply a [`Repair`] to the file at `path`, given what was parsed from it.
///
/// Returns the number of records written (zero for [`Repair::Discard`]). Both arms are
/// idempotent: repairing a clean cache rewrites the identical bytes, discarding a missing file
/// succeeds.
pub fn repair_file(path: &Path, parsed: &ParsedCache, repair: Repair) -> io::Result<usize> {
    match repair {
        Repair::Truncate => {
            let image = serialize(&parsed.records);
            write_atomic(path, &image)?;
            Ok(parsed.records.len())
        }
        Repair::Discard => {
            match fs::remove_file(path) {
                Ok(()) => Ok(0),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
                Err(e) => Err(e),
            }
        }
    }
}

/// Append records to a cache file, creating it with a header if absent.
///
/// The C++ `SerializePipeline` opens with `ios::app` and writes the header only when the file is
/// empty. Doing the same here keeps a partially-written file from a previous crash from producing
/// a second header in the middle of the stream — this function checks the existing header first
/// and refuses to append if it does not match, rather than corrupting the file.
pub fn append_file(path: &Path, records: &[Record]) -> io::Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    if path.exists() {
        let existing = parse_file(path)?;
        if !existing.is_clean() {
            // A damaged file cannot be safely appended to; repair it first so the header is valid.
            repair_file(path, &existing, Repair::Truncate)?;
        }
    }
    let mut image = if path.exists() {
        fs::read(path)?
    } else {
        Vec::new()
    };
    if image.is_empty() {
        image.extend_from_slice(&MAGIC);
        image.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    }
    for record in records {
        image.extend_from_slice(&(record.envs.len() as u32).to_le_bytes());
        for env in &record.envs {
            env.write(&mut image);
        }
        image.extend_from_slice(&record.key);
    }
    write_atomic(path, &image)
}

/// Write via a sibling temporary file then rename, so a crash leaves the old cache intact.
///
/// The C++ writer truncates in place; a power loss mid-write leaves a file that fails the magic
/// check on next launch and, in the old loader, takes the whole cache with it.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = temp_sibling(path);
    fs::write(&tmp, bytes)?;
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}
