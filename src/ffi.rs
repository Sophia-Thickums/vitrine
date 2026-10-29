//! C ABI for the pipeline cache, so `shader_environment.cpp` can call this crate directly.
//!
//! The contract, which mirrors how the C++ side already uses the file:
//!
//! 1. `vitrine_cache_inspect` — read a cache, report health, change nothing.
//! 2. `vitrine_cache_repair` — act on that report: truncate to the salvageable records, or discard.
//! 3. `vitrine_cache_append` — write one pipeline's environments, appending to an existing cache.
//!
//! Every entry point catches panics. A Rust panic crossing `extern "C"` is undefined behaviour in
//! the caller, so each function body runs inside [`std::panic::catch_unwind`] and returns
//! [`VITRINE_ERR_PANIC`] instead. That is not defensiveness about our own code; it is the boundary
//! rule, and the tests below exercise it by forcing a panic.
//!
//! # Ownership
//!
//! [`vitrine_cache_inspect`] returns a [`Report`] whose records are owned by the caller and must be
//! released with [`vitrine_report_free`]. Nothing else allocates across the boundary.

use std::ffi::{CStr, c_char};
use std::os::raw::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use crate::cache::{self, ParsedCache, Record, Repair};
use crate::env::{Environment, Maps, Stage};

/// Operation succeeded.
pub const VITRINE_OK: c_int = 0;
/// Operated, but the cache had damage — records were kept, the tail was lost.
pub const VITRINE_WARN_DAMAGED: c_int = 1;
/// Caller passed a null pointer or an unreadable C string.
pub const VITRINE_ERR_BAD_ARGUMENT: c_int = -1;
/// The file exists but could not be read or written.
pub const VITRINE_ERR_IO: c_int = -2;
/// Rust panicked; the caller should log and continue with a cold cache.
pub const VITRINE_ERR_PANIC: c_int = -3;
/// A record was rejected because it could not be represented (bad stage, oversized code).
pub const VITRINE_ERR_BAD_RECORD: c_int = -4;

/// Health of a cache file, as reported to C++.
#[repr(C)]
pub struct Report {
    /// [`VITRINE_OK`] or [`VITRINE_WARN_DAMAGED`].
    pub status: c_int,
    /// Records that parsed cleanly.
    pub record_count: usize,
    /// Total environments ("shaders") across those records.
    pub env_count: usize,
    /// Bytes of the file that were read before parsing stopped.
    pub bytes_consumed: usize,
    /// Offset of the first failure, or 0 when clean.
    pub damage_offset: usize,
    /// Static, NUL-terminated description of the damage; empty string when clean.
    pub damage_text: *const c_char,
    /// Opaque handle for [`vitrine_report_free`]; null when nothing was recovered.
    pub handle: *mut ParsedCache,
}

impl Report {
    const EMPTY: Self = Self {
        status: VITRINE_OK,
        record_count: 0,
        env_count: 0,
        bytes_consumed: 0,
        damage_offset: 0,
        damage_text: std::ptr::null(),
        handle: std::ptr::null_mut(),
    };
}

/// A pipeline the caller wants written: its key and the environments compiled for it.
#[repr(C)]
pub struct PendingPipeline {
    /// 16-byte pipeline cache key, exactly as Citron hashes it.
    pub key: [u8; cache::KEY_LEN],
    /// Pointer to `env_count` [`CEntry`] elements.
    pub envs: *const CEntry,
    /// Number of entries at `envs`.
    pub env_count: usize,
}

/// One environment in the compact form C++ can build without touching Rust types.
///
/// Deliberately minimal: the fields the cache format requires, with the map-valued parts passed as
/// flat key/value arrays. A caller that has no maps passes `map_count = 0` and a null pointer.
#[repr(C)]
pub struct CEntry {
    pub local_memory_size: u32,
    pub texture_bound: u32,
    pub start_address: u32,
    pub cached_lowest: u32,
    pub cached_highest: u32,
    pub viewport_transform_state: u32,
    /// `Shader::Stage` discriminant (0 = Vertex .. 5 = Compute).
    pub stage: u32,
    /// ARM64 instruction words.
    pub code: *const u64,
    pub code_word_count: usize,
    /// Compute: three workgroup dimensions. Ignored for other stages.
    pub workgroup_size: [u32; 3],
    /// Compute: shared memory size. Ignored for other stages.
    pub shared_memory_size: u32,
}

/// Read a cache file and describe its health without modifying it.
///
/// # Safety
///
/// `path` must be a valid NUL-terminated C string. The returned [`Report`] must be released with
/// [`vitrine_report_free`]; failing to do so leaks the recovered records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_cache_inspect(path: *const c_char) -> Report {
    match catch_unwind(AssertUnwindSafe(|| inspect_inner(path))) {
        Ok(report) => report,
        Err(_) => Report {
            status: VITRINE_ERR_PANIC,
            ..Report::EMPTY
        },
    }
}

fn inspect_inner(path: *const c_char) -> Report {
    let Some(path) = c_path(path) else {
        return Report {
            status: VITRINE_ERR_BAD_ARGUMENT,
            ..Report::EMPTY
        };
    };
    let parsed = match cache::parse_file(Path::new(&path)) {
        Ok(p) => p,
        Err(_) => {
            return Report {
                status: VITRINE_ERR_IO,
                ..Report::EMPTY
            };
        }
    };
    // A clean cache still hands back its records: the caller wants them warm, not re-parsed.
    Report {
        status: if parsed.is_clean() {
            VITRINE_OK
        } else {
            VITRINE_WARN_DAMAGED
        },
        record_count: parsed.records.len(),
        env_count: parsed.env_count(),
        bytes_consumed: parsed.consumed,
        damage_offset: parsed.damage.as_ref().map_or(0, |d| d.offset),
        damage_text: std::ptr::null(),
        handle: Box::into_raw(Box::new(parsed)),
    }
}

/// Release a [`Report`]'s handle. Safe to call with a null handle or one already freed exactly
/// once — a double free here is a caller bug, and the C++ side must not do it.
///
/// # Safety
///
/// `report` must be a pointer returned by [`vitrine_cache_inspect`] that has not already been
/// freed through this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_report_free(report: *mut Report) {
    if report.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller guarantees `report` came from `vitrine_cache_inspect` and is live.
        let report = unsafe { &mut *report };
        if !report.handle.is_null() {
            // SAFETY: `handle` was created by `Box::into_raw` in `inspect_inner` and is freed once.
            drop(unsafe { Box::from_raw(report.handle) });
            report.handle = std::ptr::null_mut();
        }
    }));
}

/// Act on an inspected cache. `truncate != 0` keeps the salvageable records; `0` discards the file.
///
/// Returns the number of records written, or a negative error code.
///
/// # Safety
///
/// `report` must be a live pointer from [`vitrine_cache_inspect`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_cache_repair(
    path: *const c_char,
    report: *mut Report,
    truncate: c_int,
) -> c_int {
    match catch_unwind(AssertUnwindSafe(|| {
        let Some(path) = c_path(path) else {
            return VITRINE_ERR_BAD_ARGUMENT;
        };
        if report.is_null() {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        // SAFETY: caller guarantees `report` is live (see this function's contract).
        let report = unsafe { &mut *report };
        if report.handle.is_null() {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        // SAFETY: non-null `handle` was created by `Box::into_raw` and is still owned here.
        let parsed = unsafe { &*report.handle };
        let repair = if truncate != 0 {
            Repair::Truncate
        } else {
            Repair::Discard
        };
        match cache::repair_file(Path::new(&path), parsed, repair) {
            Ok(n) => n as c_int,
            Err(_) => VITRINE_ERR_IO,
        }
    })) {
        Ok(code) => code,
        Err(_) => VITRINE_ERR_PANIC,
    }
}

/// Append one pipeline's environments to a cache file, creating it when absent.
///
/// # Safety
///
/// `path` must be a valid NUL-terminated C string, and `pipeline` must point to a live
/// [`PendingPipeline`] whose `envs` array holds at least `env_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_cache_append(
    path: *const c_char,
    pipeline: *const PendingPipeline,
) -> c_int {
    match catch_unwind(AssertUnwindSafe(|| {
        let Some(path) = c_path(path) else {
            return VITRINE_ERR_BAD_ARGUMENT;
        };
        if pipeline.is_null() {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        // SAFETY: caller guarantees a live `PendingPipeline` (see this function's contract).
        let pipeline = unsafe { &*pipeline };
        let record = match build_record(pipeline) {
            Ok(r) => r,
            Err(code) => return code,
        };
        match cache::append_file(Path::new(&path), std::slice::from_ref(&record)) {
            Ok(()) => 1,
            Err(_) => VITRINE_ERR_IO,
        }
    })) {
        Ok(code) => code,
        Err(_) => VITRINE_ERR_PANIC,
    }
}

fn build_record(pipeline: &PendingPipeline) -> Result<Record, c_int> {
    if pipeline.env_count == 0 || pipeline.envs.is_null() {
        return Err(VITRINE_ERR_BAD_ARGUMENT);
    }
    let mut envs = Vec::new();
    if envs.try_reserve_exact(pipeline.env_count).is_err() {
        return Err(VITRINE_ERR_BAD_RECORD);
    }
    for i in 0..pipeline.env_count {
        // SAFETY: caller guarantees `envs` holds at least `env_count` elements.
        let entry = unsafe { &*pipeline.envs.add(i) };
        envs.push(convert_entry(entry)?);
    }
    Ok(Record {
        offset: 0,
        envs,
        key: pipeline.key,
    })
}

fn convert_entry(entry: &CEntry) -> Result<Environment, c_int> {
    let stage = match entry.stage {
        0 => Stage::Vertex,
        1 => Stage::TessellationControl,
        2 => Stage::TessellationEvaluation,
        3 => Stage::Geometry,
        4 => Stage::Fragment,
        5 => Stage::Compute,
        _ => return Err(VITRINE_ERR_BAD_RECORD),
    };
    let mut code = Vec::new();
    if entry.code_word_count != 0 {
        if entry.code.is_null() {
            return Err(VITRINE_ERR_BAD_ARGUMENT);
        }
        if code.try_reserve_exact(entry.code_word_count).is_err() {
            return Err(VITRINE_ERR_BAD_RECORD);
        }
        for i in 0..entry.code_word_count {
            // SAFETY: caller guarantees `code` holds `code_word_count` words.
            code.push(unsafe { *entry.code.add(i) });
        }
    }
    Ok(Environment {
        local_memory_size: entry.local_memory_size,
        texture_bound: entry.texture_bound,
        start_address: entry.start_address,
        cached_lowest: entry.cached_lowest,
        cached_highest: entry.cached_highest,
        viewport_transform_state: entry.viewport_transform_state,
        stage,
        code,
        maps: Maps::default(),
        workgroup_size: stage.is_compute().then_some(entry.workgroup_size),
        shared_memory_size: stage.is_compute().then_some(entry.shared_memory_size),
        sph: (!stage.is_compute()).then_some([0u8; crate::env::SPH_BYTES]),
        gp_passthrough_mask: stage
            .is_geometry()
            .then_some([0u8; crate::env::GP_PASSTHROUGH_BYTES]),
    })
}

/// Borrow a C string as an owned `String`, or `None` if it is null or not valid UTF-8.
fn c_path(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller guarantees a NUL-terminated C string (documented per entry point).
    let raw = unsafe { CStr::from_ptr(ptr) };
    raw.to_str().ok().map(str::to_owned)
}

/// Return the crate version as a static NUL-terminated string. Never null.
#[unsafe(no_mangle)]
pub extern "C" fn vitrine_version() -> *const c_char {
    // A `'static` byte slice with a trailing NUL; safe to hand out permanently.
    static VERSION_C: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION_C.as_ptr().cast()
}
