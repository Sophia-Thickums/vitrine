//! FFI tests: the C ABI behaves as `shader_environment.cpp` will call it.
//!
//! These are written from the C caller's point of view — raw pointers, C strings, error codes —
//! because that is the code that will actually run against this library. A test that only uses
//! the safe Rust API would prove nothing about the boundary.

use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};

use vitrine::cache::{self, KEY_LEN, Record};
use vitrine::env::{Environment, Maps, Stage, SPH_BYTES};
use vitrine::ffi::{
    CEntry, PendingPipeline, Report, VITRINE_ERR_BAD_ARGUMENT, VITRINE_OK, VITRINE_WARN_DAMAGED,
    vitrine_cache_append, vitrine_cache_inspect, vitrine_cache_repair, vitrine_report_free,
    vitrine_version,
};

fn tmpdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vitrine-ffi-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn fixture_env() -> Environment {
    Environment {
        local_memory_size: 0,
        texture_bound: 0,
        start_address: 0x40,
        cached_lowest: 0,
        cached_highest: 0,
        viewport_transform_state: 1,
        stage: Stage::Fragment,
        code: vec![0xd503201f; 4],
        maps: Maps::default(),
        workgroup_size: None,
        shared_memory_size: None,
        sph: Some([0u8; SPH_BYTES]),
        gp_passthrough_mask: None,
    }
}

fn cstr(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

#[test]
fn version_is_a_valid_c_string() {
    let ptr = vitrine_version();
    assert!(!ptr.is_null());
    // SAFETY: `vitrine_version` returns a pointer to a static NUL-terminated string.
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
    assert_eq!(s, vitrine::VERSION);
}

#[test]
fn inspect_reports_a_clean_cache() {
    let path = tmpdir().join("ffi-clean.bin");
    let record = Record { offset: 0, envs: vec![fixture_env()], key: [1; KEY_LEN] };
    fs::write(&path, cache::serialize(&[record])).unwrap();
    let c = cstr(&path);

    // SAFETY: `c` is a valid NUL-terminated string for the duration of the call.
    let mut report = unsafe { vitrine_cache_inspect(c.as_ptr()) };
    assert_eq!(report.status, VITRINE_OK);
    assert_eq!(report.record_count, 1);
    assert_eq!(report.env_count, 1);
    assert!(report.damage_offset == 0);
    // SAFETY: `report` came from `vitrine_cache_inspect` and has not been freed.
    unsafe { vitrine_report_free(&mut report) };
}

#[test]
fn inspect_reports_damage_with_an_offset() {
    let path = tmpdir().join("ffi-damaged.bin");
    let record = Record { offset: 0, envs: vec![fixture_env()], key: [1; KEY_LEN] };
    let mut bytes = cache::serialize(&[record.clone(), record]);
    bytes.truncate(bytes.len() - 5);
    fs::write(&path, &bytes).unwrap();
    let c = cstr(&path);

    // SAFETY: valid C string.
    let mut report = unsafe { vitrine_cache_inspect(c.as_ptr()) };
    assert_eq!(report.status, VITRINE_WARN_DAMAGED);
    assert_eq!(report.record_count, 1);
    assert!(report.damage_offset > 0, "offset must point at the failure");
    // SAFETY: owned handle, freed once.
    unsafe { vitrine_report_free(&mut report) };
}

#[test]
fn inspect_rejects_a_null_path_without_crashing() {
    // SAFETY: the contract says a null pointer is an error, not UB in the caller.
    let report = unsafe { vitrine_cache_inspect(std::ptr::null()) };
    assert_eq!(report.status, VITRINE_ERR_BAD_ARGUMENT);
    assert!(report.handle.is_null());
}

#[test]
fn inspect_on_a_missing_file_is_clean_and_empty() {
    let path = tmpdir().join("ffi-missing.bin");
    let c = cstr(&path);
    // SAFETY: valid C string.
    let mut report = unsafe { vitrine_cache_inspect(c.as_ptr()) };
    assert_eq!(report.status, VITRINE_OK);
    assert_eq!(report.record_count, 0);
    // SAFETY: owned handle, freed once.
    unsafe { vitrine_report_free(&mut report) };
}

#[test]
fn repair_through_the_abi_keeps_the_good_records() {
    let path = tmpdir().join("ffi-repair.bin");
    let record = Record { offset: 0, envs: vec![fixture_env()], key: [2; KEY_LEN] };
    let mut bytes = cache::serialize(&[record.clone(), record]);
    bytes.truncate(bytes.len() - 5);
    fs::write(&path, &bytes).unwrap();
    let c = cstr(&path);

    // SAFETY: valid C string.
    let mut report = unsafe { vitrine_cache_inspect(c.as_ptr()) };
    assert_eq!(report.record_count, 1);
    // SAFETY: `report` is live and owns its handle.
    let written = unsafe { vitrine_cache_repair(c.as_ptr(), &mut report, 1) };
    assert_eq!(written, 1);
    // SAFETY: owned handle, freed once.
    unsafe { vitrine_report_free(&mut report) };

    let after = cache::parse_file(&path).unwrap();
    assert!(after.is_clean(), "repair through the ABI must leave a valid file");
    assert_eq!(after.records.len(), 1);
}

#[test]
fn append_through_the_abi_writes_a_readable_pipeline() {
    let path = tmpdir().join("ffi-append.bin");
    let c = cstr(&path);

    let code = [0xd503201fu64; 4];
    let entry = CEntry {
        local_memory_size: 0,
        texture_bound: 0,
        start_address: 0x100,
        cached_lowest: 0,
        cached_highest: 0,
        viewport_transform_state: 1,
        stage: 4, // Fragment
        code: code.as_ptr(),
        code_word_count: code.len(),
        workgroup_size: [0; 3],
        shared_memory_size: 0,
    };
    let pipeline = PendingPipeline {
        key: [7u8; KEY_LEN],
        envs: &entry,
        env_count: 1,
    };

    // SAFETY: valid C string; `pipeline` points at a live struct whose `envs` holds one element.
    let code_status = unsafe { vitrine_cache_append(c.as_ptr(), &pipeline) };
    assert_eq!(code_status, 1);

    let parsed = cache::parse_file(&path).unwrap();
    assert!(parsed.is_clean(), "{:?}", parsed.damage);
    assert_eq!(parsed.records.len(), 1);
    assert_eq!(parsed.records[0].key, [7u8; KEY_LEN]);
    assert_eq!(parsed.records[0].envs[0].code, vec![0xd503201fu64; 4]);
    assert_eq!(parsed.records[0].envs[0].stage, Stage::Fragment);
}

#[test]
fn append_rejects_null_arguments_with_an_error_code() {
    let path = tmpdir().join("ffi-null.bin");
    let c = cstr(&path);
    // SAFETY: null pipeline is documented as an error.
    assert_eq!(unsafe { vitrine_cache_append(c.as_ptr(), std::ptr::null()) }, VITRINE_ERR_BAD_ARGUMENT);
    // SAFETY: null path is documented as an error.
    let entry = CEntry {
        local_memory_size: 0, texture_bound: 0, start_address: 0, cached_lowest: 0,
        cached_highest: 0, viewport_transform_state: 1, stage: 4,
        code: std::ptr::null(), code_word_count: 0, workgroup_size: [0; 3], shared_memory_size: 0,
    };
    let pipeline = PendingPipeline { key: [0; KEY_LEN], envs: &entry, env_count: 1 };
    // SAFETY: null path is documented as an error.
    assert_eq!(unsafe { vitrine_cache_append(std::ptr::null(), &pipeline) }, VITRINE_ERR_BAD_ARGUMENT);
}

#[test]
fn report_free_tolerates_null() {
    // SAFETY: the function documents that null is a no-op.
    unsafe { vitrine_report_free(std::ptr::null_mut()) };
}

#[test]
fn an_empty_report_has_no_handle() {
    let r = Report {
        status: VITRINE_OK,
        record_count: 0,
        env_count: 0,
        bytes_consumed: 0,
        damage_offset: 0,
        damage_text: std::ptr::null(),
        handle: std::ptr::null_mut(),
    };
    assert!(r.handle.is_null());
}

// -- content selection over the C ABI --------------------------------------

use vitrine::ffi::{
    CContentEntry, CReject, CSelectResult, CWarning, VITRINE_REJECT_SUPERSEDED,
    VITRINE_TYPE_APPLICATION, VITRINE_TYPE_PATCH, VITRINE_WARN_ORPHAN_DLC,
    VITRINE_TYPE_ADD_ON_CONTENT, vitrine_content_select,
};

const BASE: u64 = 0x0100_0000_0001_0000;
const PATCH: u64 = BASE + 0x800;

#[test]
fn selection_through_the_abi_picks_the_highest_version() {
    let entries = [
        CContentEntry { title_id: BASE, version: 0, title_type: VITRINE_TYPE_APPLICATION, source: 1 },
        CContentEntry { title_id: PATCH, version: 0x20000, title_type: VITRINE_TYPE_PATCH, source: 2 },
        CContentEntry { title_id: PATCH, version: 0x30000, title_type: VITRINE_TYPE_PATCH, source: 3 },
    ];
    let mut chosen = vec![CContentEntry { title_id: 0, version: 0, title_type: 0, source: 0 }; 8];
    let mut rejects = Vec::<CReject>::new();
    let mut warnings = Vec::<CWarning>::new();
    let mut out = CSelectResult { chosen_count: 0, rejected_count: 0, warning_count: 0, truncated: 0 };

    // SAFETY: entries is live; the output arrays have the capacities passed.
    let n = unsafe {
        vitrine_content_select(
            entries.as_ptr(), entries.len(),
            chosen.as_mut_ptr(), chosen.len(),
            rejects.as_mut_ptr(), 0,
            warnings.as_mut_ptr(), 0,
            &mut out,
        )
    };
    assert_eq!(n, 2);
    assert_eq!(out.chosen_count, 2);
    assert_eq!(out.rejected_count, 1, "the older patch is superseded");
    assert_eq!(out.truncated, 1, "zero-cap rejects array must report truncation");

    let patch = chosen
        .iter()
        .find(|e| e.title_type == VITRINE_TYPE_PATCH)
        .expect("a patch must be chosen");
    assert_eq!(patch.version, 0x30000);
}

#[test]
fn title_type_crosses_the_boundary_as_the_cnmt_byte_not_a_variant_index() {
    // Regression test for a real bug: `enum as u32` yields 0..=3, not 0x80..=0x83, so the C side
    // saw every entry as unknown and selection silently returned nothing.
    let entries = [
        CContentEntry { title_id: BASE, version: 0, title_type: VITRINE_TYPE_APPLICATION, source: 1 },
        CContentEntry { title_id: PATCH, version: 5, title_type: VITRINE_TYPE_PATCH, source: 2 },
    ];
    let mut chosen = vec![CContentEntry { title_id: 0, version: 0, title_type: 0, source: 0 }; 4];
    let mut out = CSelectResult { chosen_count: 0, rejected_count: 0, warning_count: 0, truncated: 0 };
    let mut rej = Vec::<CReject>::new();
    let mut warn = Vec::<CWarning>::new();
    // SAFETY: entries live; outputs sized.
    let n = unsafe {
        vitrine_content_select(
            entries.as_ptr(), entries.len(),
            chosen.as_mut_ptr(), chosen.len(),
            rej.as_mut_ptr(), 0,
            warn.as_mut_ptr(), 0,
            &mut out,
        )
    };
    assert_eq!(n, 2, "both entries must be recognisable");
    let types: Vec<u32> = chosen[..out.chosen_count].iter().map(|e| e.title_type).collect();
    assert!(types.contains(&0x80), "application byte must survive: {types:?}");
    assert!(types.contains(&0x81), "patch byte must survive: {types:?}");
}

#[test]
fn rejects_carry_the_version_that_won() {
    let entries = [
        CContentEntry { title_id: PATCH, version: 0x10000, title_type: VITRINE_TYPE_PATCH, source: 1 },
        CContentEntry { title_id: PATCH, version: 0x90000, title_type: VITRINE_TYPE_PATCH, source: 2 },
    ];
    let mut chosen = vec![CContentEntry { title_id: 0, version: 0, title_type: 0, source: 0 }; 4];
    let mut rejects = vec![CReject {
        entry: CContentEntry { title_id: 0, version: 0, title_type: 0, source: 0 },
        reason: 99,
        chosen_version: 0,
    }; 4];
    let mut warnings = Vec::<CWarning>::new();
    let mut out = CSelectResult { chosen_count: 0, rejected_count: 0, warning_count: 0, truncated: 0 };
    // SAFETY: all arrays live with the stated capacities.
    unsafe {
        vitrine_content_select(
            entries.as_ptr(), entries.len(),
            chosen.as_mut_ptr(), chosen.len(),
            rejects.as_mut_ptr(), rejects.len(),
            warnings.as_mut_ptr(), 0,
            &mut out,
        )
    };
    assert_eq!(out.rejected_count, 1);
    assert_eq!(rejects[0].reason, VITRINE_REJECT_SUPERSEDED);
    assert_eq!(rejects[0].chosen_version, 0x90000);
}

#[test]
fn orphan_dlc_warning_crosses_the_abi() {
    let dlc = BASE + 0x1000;
    let entries = [CContentEntry {
        title_id: dlc,
        version: 0,
        title_type: VITRINE_TYPE_ADD_ON_CONTENT,
        source: 7,
    }];
    let mut chosen = vec![CContentEntry { title_id: 0, version: 0, title_type: 0, source: 0 }; 2];
    let mut rejects = Vec::<CReject>::new();
    let mut warnings = vec![CWarning { kind: 99, title_id: 0, version: 0 }; 2];
    let mut out = CSelectResult { chosen_count: 0, rejected_count: 0, warning_count: 0, truncated: 0 };
    // SAFETY: all arrays live with the stated capacities.
    unsafe {
        vitrine_content_select(
            entries.as_ptr(), entries.len(),
            chosen.as_mut_ptr(), chosen.len(),
            rejects.as_mut_ptr(), 0,
            warnings.as_mut_ptr(), warnings.len(),
            &mut out,
        )
    };
    assert_eq!(out.warning_count, 1);
    assert_eq!(warnings[0].kind, VITRINE_WARN_ORPHAN_DLC);
    assert_eq!(warnings[0].title_id, dlc);
}

#[test]
fn selection_rejects_null_pointers() {
    let mut out = CSelectResult { chosen_count: 0, rejected_count: 0, warning_count: 0, truncated: 0 };
    // SAFETY: null entries is documented as an error, not a crash.
    let code = unsafe {
        vitrine_content_select(
            std::ptr::null(), 1,
            std::ptr::null_mut(), 0,
            std::ptr::null_mut(), 0,
            std::ptr::null_mut(), 0,
            &mut out,
        )
    };
    assert_eq!(code, vitrine::ffi::VITRINE_ERR_BAD_ARGUMENT);
}

#[test]
fn selection_of_nothing_is_zero_and_clean() {
    let mut out = CSelectResult { chosen_count: 9, rejected_count: 9, warning_count: 9, truncated: 9 };
    // SAFETY: a null entries pointer with count 0 is rejected by the null check, so pass a real one.
    let empty: [CContentEntry; 0] = [];
    let code = unsafe {
        vitrine_content_select(
            empty.as_ptr(), 0,
            std::ptr::null_mut(), 0,
            std::ptr::null_mut(), 0,
            std::ptr::null_mut(), 0,
            &mut out,
        )
    };
    assert_eq!(code, 0);
    assert_eq!(out.chosen_count, 0);
    assert_eq!(out.rejected_count, 0);
    assert_eq!(out.truncated, 0);
}
