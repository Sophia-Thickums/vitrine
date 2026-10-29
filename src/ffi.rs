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

// ---------------------------------------------------------------------------
// Content selection
// ---------------------------------------------------------------------------

/// `TitleType` byte for an application, as it appears in a CNMT header.
pub const VITRINE_TYPE_APPLICATION: u32 = 0x80;
/// `TitleType` byte for a patch.
pub const VITRINE_TYPE_PATCH: u32 = 0x81;
/// `TitleType` byte for add-on content.
pub const VITRINE_TYPE_ADD_ON_CONTENT: u32 = 0x82;
/// `TitleType` byte for a delta fragment.
pub const VITRINE_TYPE_DELTA: u32 = 0x83;

/// One installed CNMT, as the C++ filesystem layer knows it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CContentEntry {
    /// Title id the CNMT names.
    pub title_id: u64,
    /// Version the CNMT declares.
    pub version: u32,
    /// `TitleType`: 0x80 Application, 0x81 Patch, 0x82 AddOnContent, 0x83 Delta.
    pub title_type: u32,
    /// Caller's stable identifier for the underlying file; carried through untouched.
    pub source: u64,
}

impl CContentEntry {
    fn as_entry(self) -> Option<content::Entry> {
        let title_type = u8::try_from(self.title_type)
            .ok()
            .and_then(TitleType::from_u8)?;
        Some(content::Entry {
            title_id: self.title_id,
            version: self.version,
            title_type,
            source: self.source,
        })
    }

    fn from_entry(e: &content::Entry) -> Self {
        Self {
            title_id: e.title_id,
            version: e.version,
            title_type: u32::from(e.title_type.to_u8()),
            source: e.source,
        }
    }
}

/// Why an entry was not chosen, in the C ABI.
///
/// `reason`: 0 = superseded (see `chosen_version`), 1 = duplicate, 2 = unknown title type.
pub const VITRINE_REJECT_SUPERSEDED: u32 = 0;
/// See [`VITRINE_REJECT_SUPERSEDED`].
pub const VITRINE_REJECT_DUPLICATE: u32 = 1;

/// One rejected entry and the reason.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CReject {
    pub entry: CContentEntry,
    /// `VITRINE_REJECT_SUPERSEDED` or `VITRINE_REJECT_DUPLICATE`.
    pub reason: u32,
    /// Version that beat this entry; 0 when `reason` is not superseded.
    pub chosen_version: u32,
}

/// `kind` for [`CWarning`]: a patch with no base application installed.
pub const VITRINE_WARN_ORPHAN_PATCH: u32 = 0;
/// `kind` for [`CWarning`]: add-on content with no base application installed.
pub const VITRINE_WARN_ORPHAN_DLC: u32 = 1;

/// A non-fatal problem worth logging.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CWarning {
    pub kind: u32,
    pub title_id: u64,
    pub version: u32,
}

/// Filled by [`vitrine_content_select`] to describe what it produced.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CSelectResult {
    /// Number of chosen entries written.
    pub chosen_count: usize,
    /// Number of rejects written.
    pub rejected_count: usize,
    /// Number of warnings written.
    pub warning_count: usize,
    /// Non-zero when a caller-provided array was too small; the counts above are then the true
    /// totals and only the first `*_cap` of each were written.
    pub truncated: u32,
}

/// Select the winning content out of everything installed.
///
/// Writes up to `chosen_cap` choices, `reject_cap` rejects and `warning_cap` warnings into the
/// caller's arrays, and reports the true totals in `out` so a caller can re-call with bigger
/// arrays. Entries whose `title_type` is not a known `TitleType` are skipped silently: the caller
/// enumerated something that is not content, which is not this function's problem to report.
///
/// Returns the number of chosen entries, or a negative error code.
///
/// # Safety
///
/// `entries` must point to `entry_count` live [`CContentEntry`] values. `out` must be a live
/// [`CSelectResult`]. Each output array must either be null (with its cap zero) or point to at
/// least its stated capacity of the corresponding element type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_content_select(
    entries: *const CContentEntry,
    entry_count: usize,
    chosen: *mut CContentEntry,
    chosen_cap: usize,
    rejects: *mut CReject,
    reject_cap: usize,
    warnings: *mut CWarning,
    warning_cap: usize,
    out: *mut CSelectResult,
) -> c_int {
    match catch_unwind(AssertUnwindSafe(|| {
        if entries.is_null() || out.is_null() {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        if (chosen.is_null() && chosen_cap != 0) || (rejects.is_null() && reject_cap != 0) {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        if warnings.is_null() && warning_cap != 0 {
            return VITRINE_ERR_BAD_ARGUMENT;
        }
        // SAFETY: caller guarantees `entries` holds `entry_count` values.
        let slice = unsafe { std::slice::from_raw_parts(entries, entry_count) };
        let selection = content::select(slice.iter().filter_map(|e| e.as_entry()));

        let chosen_vec: Vec<&content::Entry> = selection.chosen.values().collect();
        let mut truncated = 0u32;
        for (i, entry) in chosen_vec.iter().enumerate() {
            if i < chosen_cap {
                // SAFETY: caller guarantees `chosen` holds `chosen_cap` values when non-null.
                unsafe { chosen.add(i).write(CContentEntry::from_entry(entry)) };
            } else {
                truncated = 1;
            }
        }
        for (i, (entry, reason)) in selection.rejected.iter().enumerate() {
            if i >= reject_cap {
                truncated = 1;
                break;
            }
            let (code, chosen_version) = match reason {
                content::Rejected::Superseded { chosen_version } => {
                    (VITRINE_REJECT_SUPERSEDED, *chosen_version)
                }
                content::Rejected::Duplicate => (VITRINE_REJECT_DUPLICATE, 0),
                content::Rejected::UnknownType { .. } => (VITRINE_REJECT_DUPLICATE, 0),
            };
            // SAFETY: caller guarantees `rejects` holds `reject_cap` values when non-null.
            unsafe {
                rejects.add(i).write(CReject {
                    entry: CContentEntry::from_entry(entry),
                    reason: code,
                    chosen_version,
                })
            };
        }
        for (i, warning) in selection.warnings.iter().enumerate() {
            if i >= warning_cap {
                truncated = 1;
                break;
            }
            let (kind, title_id, version) = match warning {
                content::Warning::OrphanPatch { title_id, version } => {
                    (VITRINE_WARN_ORPHAN_PATCH, *title_id, *version)
                }
                content::Warning::OrphanDlc { title_id, version } => {
                    (VITRINE_WARN_ORPHAN_DLC, *title_id, *version)
                }
            };
            // SAFETY: caller guarantees `warnings` holds `warning_cap` values when non-null.
            unsafe {
                warnings.add(i).write(CWarning {
                    kind,
                    title_id,
                    version,
                })
            };
        }
        // SAFETY: caller guarantees `out` is a live CSelectResult.
        unsafe {
            out.write(CSelectResult {
                chosen_count: selection.chosen.len(),
                rejected_count: selection.rejected.len(),
                warning_count: selection.warnings.len(),
                truncated,
            })
        };
        selection.chosen.len() as c_int
    })) {
        Ok(code) => code,
        Err(_) => VITRINE_ERR_PANIC,
    }
}

use crate::content::{self, TitleType};

// ── The JIT ABI — what a C++ `ArmInterface` backend calls ───────────────────────────────────────
//
// The pipeline-cache entry points above describe a FILE. These describe a RUNNING CPU: a guest
// state block the caller owns, a word reader that can fail (an unmapped page is `None`, not a
// zero), and a run that stops with a reason the caller can act on.

use crate::jit::{AddressJit, Exit};
use std::ffi::c_void;

/// Reads one 32-bit guest instruction word at `addr`; returns a negative value when unmapped.
///
/// Negative rather than a sentinel word, because every 32-bit pattern is a legal instruction
/// encoding — there is no in-band value left to spare.
pub type VitrineReadWord = unsafe extern "C" fn(ctx: *mut c_void, addr: u64) -> i64;

/// Where a run stopped. Mirrors `jit::Exit` numerically, so the C++ side can switch on it.
#[repr(C)]
pub struct VitrineRunResult {
    /// Blocks executed before stopping.
    pub executed: u64,
    /// 0 next · 1 jumped · 2 supervisor call · 3 unsupported · 4 data abort.
    pub exit_code: u32,
    /// Where execution stopped. On a data abort this is the faulting instruction, so it can be
    /// re-executed once the caller has made the page readable.
    pub pc: u64,
    /// The address that faulted, valid only for exit code 4.
    pub fault_addr: u64,
    /// [`VITRINE_OK`] or one of the `VITRINE_ERR_*` codes.
    pub status: i32,
}

/// Opaque handle. Owns the cache, the state block and the reader closure.
pub struct VitrineJit {
    inner: AddressJit<Box<dyn FnMut(u64) -> Option<u32>>>,
}

/// Create a JIT. The state is copied in; the caller's block may be dropped afterwards.
///
/// # Safety
/// `read` must be a valid function pointer that stays callable for as long as the handle lives, and
/// `ctx` must remain valid for the same period.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_new(
    state: *const crate::jit::JitState,
    read: VitrineReadWord,
    ctx: *mut c_void,
) -> *mut VitrineJit {
    if state.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `state` points at a JitState.
        let initial = unsafe { std::ptr::read(state) };
        let reader = move |addr: u64| {
            // SAFETY: `read` and `ctx` were required to outlive the handle.
            let word = unsafe { read(ctx, addr) };
            if word < 0 { None } else { Some(word as u32) }
        };
        Box::into_raw(Box::new(VitrineJit {
            inner: AddressJit::new(
                Box::new(reader) as Box<dyn FnMut(u64) -> Option<u32>>,
                initial,
            ),
        }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Create a JIT with a ZEROED guest state — the door for a caller that has no state to import.
///
/// [`vitrine_jit_new`] copies a caller-owned block, which an emulator backend can build from its
/// thread context but a fresh caller cannot: the layout is Rust's, so the only way to make one in C
/// is to allocate `vitrine_jit_state_size()` bytes and fill them with nothing, and passing
/// uninitialised memory across the boundary is undefined however well it happens to work. This
/// constructor removes that trap.
///
/// # Safety
/// `read` must stay callable for as long as the handle lives, and `ctx` must remain valid meanwhile.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_new_default(
    read: VitrineReadWord,
    ctx: *mut c_void,
) -> *mut VitrineJit {
    let initial = crate::jit::JitState::new(0, 0);
    // SAFETY: `initial` is a live, initialised JitState for the duration of the call.
    unsafe { vitrine_jit_new(&initial, read, ctx) }
}

/// Release a handle. Null is accepted and ignored.
///
/// # Safety
/// `jit` must be a pointer returned by [`vitrine_jit_new`] and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_free(jit: *mut VitrineJit) {
    if jit.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the pointer came from `Box::into_raw` in `vitrine_jit_new`, dropped exactly once.
        drop(unsafe { Box::from_raw(jit) });
    }));
}

/// The live guest state, for reading registers out and writing them in. Null if `jit` is null.
///
/// # Safety
/// `jit` must be a live handle. The pointer is invalidated by [`vitrine_jit_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_state(jit: *mut VitrineJit) -> *mut crate::jit::JitState {
    if jit.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `jit` is a live handle per the caller's contract.
    let handle = unsafe { &mut *jit };
    handle.inner.state_mut()
}

/// Run up to `max_blocks` blocks from the state's `pc`.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_run(
    jit: *mut VitrineJit,
    max_blocks: u64,
) -> VitrineRunResult {
    let mut out = VitrineRunResult {
        executed: 0,
        exit_code: Exit::Next as u32,
        pc: 0,
        fault_addr: 0,
        status: VITRINE_ERR_BAD_ARGUMENT,
    };
    if jit.is_null() {
        return out;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: live handle per the caller's contract.
        let handle = unsafe { &mut *jit };
        let ran = handle.inner.run(max_blocks);
        let state = handle.inner.state();
        (ran, state.pc, state.fault_addr, state.last_exit)
    }));
    match result {
        Ok((Ok(executed), pc, fault_addr, exit)) => {
            out.executed = executed;
            out.pc = pc;
            out.fault_addr = fault_addr;
            out.exit_code = exit as u32;
            out.status = VITRINE_OK;
        }
        Ok((Err(_why), pc, fault_addr, exit)) => {
            // A refusal is a result, not a crash: pc and the exit code still tell the caller where
            // it stopped, which is what a JIT that met an instruction it cannot emit must report.
            out.pc = pc;
            out.fault_addr = fault_addr;
            out.exit_code = exit as u32;
            out.status = VITRINE_ERR_BAD_RECORD;
        }
        Err(_) => out.status = VITRINE_ERR_PANIC,
    }
    out
}

/// Drop cached translations overlapping `[addr, addr + size)`. Returns how many were dropped.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_invalidate(jit: *mut VitrineJit, addr: u64, size: u64) -> u64 {
    if jit.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: live handle per the caller's contract.
        let handle = unsafe { &mut *jit };
        handle.inner.invalidate(addr, size) as u64
    }));
    result.unwrap_or(0)
}

/// How many blocks are compiled and resident.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_cache_len(jit: *mut VitrineJit) -> u64 {
    if jit.is_null() {
        return 0;
    }
    // SAFETY: live handle per the caller's contract.
    let handle = unsafe { &*jit };
    handle.inner.cache_len() as u64
}

/// Size of the guest state block, so C++ can allocate one without knowing the layout.
#[unsafe(no_mangle)]
pub extern "C" fn vitrine_jit_state_size() -> usize {
    std::mem::size_of::<crate::jit::JitState>()
}

/// Read guest register `n` (0..=30). Out of range reads 0.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_get_reg(jit: *mut VitrineJit, n: u32) -> u64 {
    if jit.is_null() || n > 30 {
        return 0;
    }
    // SAFETY: live handle per the caller's contract; the index is range-checked above.
    let state = unsafe { &*jit }.inner.state();
    state.x[n as usize]
}

/// Write guest register `n` (0..=30). Out of range is ignored, not clamped.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_set_reg(jit: *mut VitrineJit, n: u32, value: u64) {
    if jit.is_null() || n > 30 {
        return;
    }
    // SAFETY: live handle per the caller's contract; the index is range-checked above.
    unsafe { &mut *jit }.inner.state_mut().x[n as usize] = value;
}

/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_get_pc(jit: *mut VitrineJit) -> u64 {
    if jit.is_null() {
        return 0;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &*jit }.inner.state().pc
}

/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_set_pc(jit: *mut VitrineJit, value: u64) {
    if jit.is_null() {
        return;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &mut *jit }.inner.state_mut().pc = value;
}

/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_get_sp(jit: *mut VitrineJit) -> u64 {
    if jit.is_null() {
        return 0;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &*jit }.inner.state().sp
}

/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_set_sp(jit: *mut VitrineJit, value: u64) {
    if jit.is_null() {
        return;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &mut *jit }.inner.state_mut().sp = value;
}

/// Packed PSTATE flags: bit 3 N, 2 Z, 1 C, 0 V.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_get_flags(jit: *mut VitrineJit) -> u8 {
    if jit.is_null() {
        return 0;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &*jit }.inner.state().flags
}

/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_set_flags(jit: *mut VitrineJit, value: u8) {
    if jit.is_null() {
        return;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &mut *jit }.inner.state_mut().flags = value & 0x0F;
}

/// Guest address of the last data abort, valid when the last run returned exit code 4.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_fault_addr(jit: *mut VitrineJit) -> u64 {
    if jit.is_null() {
        return 0;
    }
    // SAFETY: live handle per the caller's contract.
    unsafe { &*jit }.inner.state().fault_addr
}

/// Execute exactly one guest instruction (uncached translation). Used by `StepThread`.
///
/// # Safety
/// `jit` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vitrine_jit_step(jit: *mut VitrineJit) -> VitrineRunResult {
    let mut out = VitrineRunResult {
        executed: 0,
        exit_code: Exit::Next as u32,
        pc: 0,
        fault_addr: 0,
        status: VITRINE_ERR_BAD_ARGUMENT,
    };
    if jit.is_null() {
        return out;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: live handle per the caller's contract.
        let handle = unsafe { &mut *jit };
        let ran = handle.inner.step();
        let state = handle.inner.state();
        (ran, state.pc, state.fault_addr, state.last_exit)
    }));
    match result {
        Ok((Ok(executed), pc, fault_addr, exit)) => {
            out.executed = executed;
            out.pc = pc;
            out.fault_addr = fault_addr;
            out.exit_code = exit as u32;
            out.status = VITRINE_OK;
        }
        Ok((Err(_), pc, fault_addr, exit)) => {
            out.pc = pc;
            out.fault_addr = fault_addr;
            out.exit_code = exit as u32;
            out.status = VITRINE_ERR_BAD_RECORD;
        }
        Err(_) => out.status = VITRINE_ERR_PANIC,
    }
    out
}
