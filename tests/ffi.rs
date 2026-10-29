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
