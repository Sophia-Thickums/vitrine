//! Vitrine — the Rust core for the Citron Neo emulator.
//!
//! The name is the display case a specimen is kept in: this crate holds the parts of the emulator
//! that benefit from being written in a language where a mistake is a compile error rather than a
//! crash three sessions later, and hands them to the C++ engine across a narrow, `extern "C"`
//! boundary. It is a component library, not a replacement for the emulator.
//!
//! # What lives here
//!
//! - [`cache`] — the Vulkan pipeline cache reader/writer. Replaces `LoadPipelines` and
//!   `SerializePipeline`, whose current behaviour of deleting the whole cache on one bad record is
//!   the single most user-visible defect in the tracker.
//! - [`env`] — the serialized shader environment, field for field with `GenericEnvironment`, with
//!   deterministic ordering in place of `std::unordered_map` iteration.
//! - [`ffi`] — the C ABI the C++ side links against. Every function here is `unsafe` at the
//!   boundary and fully checked inside it; a caller cannot cause a crash by passing bad lengths,
//!   only a documented error code.
//!
//! # Integration
//!
//! Build with `cargo build --release` to produce `libvitrine.so` (or `.a`), add it to the Citron
//! CMake target, include `vitrine.h`, and replace the two call sites in
//! `src/video_core/shader_environment.cpp`. Nothing else in the C++ tree changes: the cache file
//! format is unchanged, so an existing user's shaders are still valid.
//!
//! # Licensing
//!
//! GPL-3.0-or-later, matching Citron Neo. A C++ project under GPL may link this crate; the
//! reverse direction, a permissive project linking this, may not.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod a64;
pub mod block;
pub mod cache;
pub mod content;
pub mod cursor;
pub mod env;
pub mod ffi;

pub use cache::{
    CACHE_VERSION, Damage, DamageKind, KEY_LEN, MAGIC, MAX_ENVS, ParsedCache, Record, Repair,
    append_file, parse, parse_file, repair_file, serialize,
};
pub use block::{Block, Exit, Program, scan};
pub use content::{Entry, Rejected, Selection, TitleType, Warning, base_title_id, select};
pub use env::{Environment, EnvParseError, Maps, Stage};

/// The crate version, exposed to C++ for logging and for `hermes doctor`-style checks.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
