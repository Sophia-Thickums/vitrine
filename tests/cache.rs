//! Integration tests: the cache reader against the behaviour it replaces.
//!
//! Each test names the C++ behaviour it pins, so that a future reader can tell whether a change
//! here is a deliberate divergence or an accident.

use std::fs;
use std::path::PathBuf;

use vitrine::cache::{
    self, CACHE_VERSION, Damage, DamageKind, KEY_LEN, MAGIC, MAX_ENVS, ParsedCache, Record, Repair,
};
use vitrine::env::{Environment, Maps, SPH_BYTES, Stage};

// -- fixtures ---------------------------------------------------------------

fn env_fragment() -> Environment {
    Environment {
        local_memory_size: 0,
        texture_bound: 0,
        start_address: 0x100,
        cached_lowest: 0,
        cached_highest: 0,
        viewport_transform_state: 1,
        stage: Stage::Fragment,
        code: vec![0xd503201f, 0xd503201f], // two NOPs
        maps: Maps::default(),
        workgroup_size: None,
        shared_memory_size: None,
        sph: Some([0u8; SPH_BYTES]),
        gp_passthrough_mask: None,
    }
}

fn compute_env() -> Environment {
    Environment {
        stage: Stage::Compute,
        workgroup_size: Some([8, 4, 1]),
        shared_memory_size: Some(1024),
        sph: None,
        ..env_fragment()
    }
}

fn geometry_env() -> Environment {
    Environment {
        stage: Stage::Geometry,
        gp_passthrough_mask: Some([7u8; 32]),
        ..env_fragment()
    }
}

fn record(n: usize) -> Record {
    Record {
        offset: 0,
        envs: vec![env_fragment()],
        key: [n as u8; KEY_LEN],
    }
}

fn header() -> Vec<u8> {
    let mut v = MAGIC.to_vec();
    v.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    v
}

fn tmpdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vitrine-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

// -- round trip -------------------------------------------------------------

#[test]
fn empty_cache_round_trips() {
    let bytes = cache::serialize(&[]);
    assert_eq!(bytes.len(), 12, "header only");
    let parsed = cache::parse(&bytes);
    assert!(parsed.is_clean());
    assert_eq!(parsed.records.len(), 0);
}

#[test]
fn records_round_trip_byte_for_byte() {
    let records = vec![record(1), record(2), record(3)];
    let bytes = cache::serialize(&records);
    let parsed = cache::parse(&bytes);
    assert!(parsed.is_clean(), "{:?}", parsed.damage);
    assert_eq!(parsed.records.len(), 3);
    // Keys survive in order; envs compare structurally.
    assert_eq!(parsed.records[0].key, [1u8; KEY_LEN]);
    assert_eq!(parsed.records[2].key, [3u8; KEY_LEN]);
    assert_eq!(parsed.records[0].envs[0], records[0].envs[0]);
    // Re-serializing a clean cache must reproduce the identical bytes.
    assert_eq!(cache::serialize(&parsed.records), bytes);
}

#[test]
fn compute_and_geometry_stages_round_trip() {
    let records = vec![
        Record {
            offset: 0,
            envs: vec![compute_env()],
            key: [9; KEY_LEN],
        },
        Record {
            offset: 0,
            envs: vec![geometry_env()],
            key: [8; KEY_LEN],
        },
    ];
    let bytes = cache::serialize(&records);
    let parsed = cache::parse(&bytes);
    assert!(parsed.is_clean(), "{:?}", parsed.damage);
    assert_eq!(parsed.records.len(), 2);
    assert_eq!(parsed.records[0].envs[0].workgroup_size, Some([8, 4, 1]));
    assert_eq!(parsed.records[0].envs[0].shared_memory_size, Some(1024));
    assert_eq!(
        parsed.records[1].envs[0].gp_passthrough_mask,
        Some([7u8; 32])
    );
    assert_eq!(cache::serialize(&parsed.records), bytes);
}

#[test]
fn multiple_envs_per_record() {
    let records = vec![Record {
        offset: 0,
        envs: vec![env_fragment(), compute_env(), geometry_env()],
        key: [5; KEY_LEN],
    }];
    let bytes = cache::serialize(&records);
    let parsed = cache::parse(&bytes);
    assert!(parsed.is_clean(), "{:?}", parsed.damage);
    assert_eq!(parsed.records[0].envs.len(), 3);
    assert_eq!(parsed.env_count(), 3);
}

#[test]
fn serialization_is_order_independent_of_map_insertion() {
    // The C++ writer emits unordered_map contents in unspecified order. Two maps with the same
    // entries must produce identical bytes here regardless of how they were built.
    let mut a = Maps::default();
    for i in [5u32, 1, 9, 3] {
        a.texture_types.insert(i, i * 2);
    }
    let mut b = Maps::default();
    for i in [9u32, 3, 5, 1] {
        b.texture_types.insert(i, i * 2);
    }
    let e1 = Environment {
        maps: a,
        ..env_fragment()
    };
    let e2 = Environment {
        maps: b,
        ..env_fragment()
    };
    assert_eq!(
        cache::serialize(&[Record {
            offset: 0,
            envs: vec![e1],
            key: [0; KEY_LEN]
        }]),
        cache::serialize(&[Record {
            offset: 0,
            envs: vec![e2],
            key: [0; KEY_LEN]
        }]),
        "BTreeMap ordering must make serialization deterministic"
    );
}

// -- damage ----------------------------------------------------------------

#[test]
fn missing_file_is_an_empty_clean_cache_not_an_error() {
    // C++ returns early when the stream fails to open; this is that behaviour, made explicit.
    let path = tmpdir().join("does-not-exist.bin");
    let parsed = cache::parse_file(&path).unwrap();
    assert!(parsed.is_clean());
    assert_eq!(parsed.records.len(), 0);
}

#[test]
fn short_header_is_reported_not_swallowed() {
    let parsed = cache::parse(b"yuzu");
    let damage = parsed.damage.expect("must report damage");
    assert_eq!(damage.offset, 0);
    assert!(matches!(damage.reason, DamageKind::ShortHeader { have: 4 }));
}

#[test]
fn bad_magic_is_reported_with_the_bytes_found() {
    let parsed = cache::parse(b"NOTACACHE\x01\x00\x00\x00");
    match parsed.damage.expect("must report damage").reason {
        DamageKind::BadMagic { found } => assert_eq!(&found, b"NOTACACH"),
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn version_mismatch_is_named_as_incompatible_not_corrupt() {
    let mut bytes = header();
    bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
    match cache::parse(&bytes).damage.expect("must report").reason {
        DamageKind::VersionMismatch { found, expected } => {
            assert_eq!(found, 99);
            assert_eq!(expected, CACHE_VERSION);
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
}

#[test]
fn zero_or_oversized_env_count_is_rejected() {
    // The C++ loader throws on `num_envs == 0 || > 64`; both are reported here.
    for bad in [0u32, MAX_ENVS + 1] {
        let mut bytes = header();
        bytes.extend_from_slice(&bad.to_le_bytes());
        let parsed = cache::parse(&bytes);
        match parsed.damage.expect("must report").reason {
            DamageKind::BadEnvCount { found } => assert_eq!(found, bad),
            other => panic!("expected BadEnvCount for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn truncated_tail_keeps_every_complete_record() {
    // The behaviour that matters: a crash mid-append must not cost the shaders before it.
    let records = vec![record(1), record(2), record(3)];
    let full = cache::serialize(&records);
    let cut = full.len() - 3; // slice into the final record
    let parsed = cache::parse(&full[..cut]);
    let damage = parsed.damage.expect("must report damage");
    assert!(matches!(damage.reason, DamageKind::Truncated { .. }));
    // Two complete records survive. This is the whole point of the rewrite: the C++ code
    // would have deleted the file here, taking all three.
    assert_eq!(parsed.records.len(), 2);
    assert_eq!(parsed.records[0].key, [1u8; KEY_LEN]);
    assert_eq!(parsed.records[1].key, [2u8; KEY_LEN]);
}

#[test]
fn corrupt_middle_record_keeps_what_came_before_it() {
    let records = vec![record(1), record(2), record(3)];
    let mut bytes = cache::serialize(&records);
    // Overwrite the num_envs field of the second record with an impossible count.
    let second_offset = parsed_offsets(&bytes)[1];
    bytes[second_offset..second_offset + 4].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
    let parsed = cache::parse(&bytes);
    assert!(!parsed.is_clean());
    assert_eq!(parsed.records.len(), 1, "only the first record is intact");
    assert_eq!(parsed.consumed, second_offset);
}

fn parsed_offsets(bytes: &[u8]) -> Vec<usize> {
    // Walk the serialized image using the same record rules the parser uses: num_envs(4) + envs
    // + key(16). We learn each env's length by serializing the fixture it came from, then skip.
    // A simpler route: parse a copy with the damaged byte restored is not possible, so we locate
    // record boundaries from the clean image the test started with.
    let _ = bytes;
    // The image is header + record + record + record with identical fixtures, so the stride is
    // (len - 12) / 3.
    let body = bytes.len() - 12;
    let stride = body / 3;
    vec![12, 12 + stride, 12 + 2 * stride]
}

#[test]
fn oversized_code_size_is_refused_before_allocation() {
    let mut bytes = header();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_envs
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // code_size
    for _ in 0..5 {
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    let parsed = cache::parse(&bytes);
    let damage = parsed.damage.expect("must report");
    match damage.reason {
        DamageKind::BadEnvironment { inner } => {
            assert!(
                matches!(
                    inner,
                    vitrine::EnvParseError::CodeSizeTooLarge { .. }
                        | vitrine::EnvParseError::BadCodeSize { .. }
                        | vitrine::EnvParseError::MapTooLarge { .. }
                ),
                "got {inner:?}"
            );
        }
        other => panic!("expected BadEnvironment, got {other:?}"),
    }
}

// -- repair ----------------------------------------------------------------

#[test]
fn repair_truncate_preserves_the_good_records_on_disk() {
    let dir = tmpdir();
    let path = dir.join("repair.bin");
    let records = vec![record(1), record(2)];
    let mut bytes = cache::serialize(&records);
    bytes.truncate(bytes.len() - 2); // damage the tail
    fs::write(&path, &bytes).unwrap();

    let parsed = cache::parse_file(&path).unwrap();
    assert!(!parsed.is_clean());
    let kept = cache::repair_file(&path, &parsed, Repair::Truncate).unwrap();
    assert_eq!(kept, 1);

    let after = cache::parse_file(&path).unwrap();
    assert!(after.is_clean(), "repair must produce a valid cache");
    assert_eq!(after.records.len(), 1);
    assert_eq!(after.records[0].key, [1u8; KEY_LEN]);
}

#[test]
fn repair_discard_removes_the_file_and_is_idempotent() {
    let dir = tmpdir();
    let path = dir.join("discard.bin");
    fs::write(&path, cache::serialize(&[record(1)])).unwrap();
    let parsed = cache::parse_file(&path).unwrap();
    assert_eq!(
        cache::repair_file(&path, &parsed, Repair::Discard).unwrap(),
        0
    );
    assert!(!path.exists());
    // Second discard on a missing file succeeds rather than erroring.
    let again = cache::parse_file(&path).unwrap();
    assert_eq!(
        cache::repair_file(&path, &again, Repair::Discard).unwrap(),
        0
    );
}

#[test]
fn repairing_a_clean_cache_changes_nothing() {
    let dir = tmpdir();
    let path = dir.join("clean.bin");
    let bytes = cache::serialize(&[record(1), record(2)]);
    fs::write(&path, &bytes).unwrap();
    let parsed = cache::parse_file(&path).unwrap();
    cache::repair_file(&path, &parsed, Repair::Truncate).unwrap();
    assert_eq!(
        fs::read(&path).unwrap(),
        bytes,
        "rewrite must be byte-identical"
    );
}

// -- append ----------------------------------------------------------------

#[test]
fn append_creates_a_file_with_a_header() {
    let dir = tmpdir();
    let path = dir.join("append-new.bin");
    cache::append_file(&path, &[record(1)]).unwrap();
    let parsed = cache::parse_file(&path).unwrap();
    assert!(parsed.is_clean());
    assert_eq!(parsed.records.len(), 1);
}

#[test]
fn append_adds_to_an_existing_file_without_a_second_header() {
    let dir = tmpdir();
    let path = dir.join("append-two.bin");
    cache::append_file(&path, &[record(1)]).unwrap();
    cache::append_file(&path, &[record(2)]).unwrap();
    let parsed = cache::parse_file(&path).unwrap();
    assert!(parsed.is_clean(), "{:?}", parsed.damage);
    assert_eq!(parsed.records.len(), 2);
    let raw = fs::read(&path).unwrap();
    assert_eq!(&raw[0..8], &MAGIC, "header intact");
    // Exactly one header: the second copy of the magic must not appear at a record boundary.
    let count = raw.windows(8).filter(|w| *w == MAGIC).count();
    assert_eq!(count, 1, "a second header would corrupt the stream");
}

#[test]
fn append_repairs_a_damaged_file_first() {
    let dir = tmpdir();
    let path = dir.join("append-damaged.bin");
    let mut bytes = cache::serialize(&[record(1), record(2)]);
    bytes.truncate(bytes.len() - 4);
    fs::write(&path, &bytes).unwrap();
    cache::append_file(&path, &[record(3)]).unwrap();
    let parsed = cache::parse_file(&path).unwrap();
    assert!(
        parsed.is_clean(),
        "append must leave a valid file: {:?}",
        parsed.damage
    );
    assert_eq!(parsed.records.len(), 2, "record 1 survived, 3 appended");
}

#[test]
fn append_of_nothing_is_a_no_op() {
    let dir = tmpdir();
    let path = dir.join("append-none.bin");
    cache::append_file(&path, &[]).unwrap();
    assert!(!path.exists(), "no records, no file");
}

// -- invariants ------------------------------------------------------------

#[test]
fn parsed_cache_reports_env_count_across_records() {
    let records = vec![
        Record {
            offset: 0,
            envs: vec![env_fragment(), env_fragment()],
            key: [1; KEY_LEN],
        },
        Record {
            offset: 0,
            envs: vec![env_fragment()],
            key: [2; KEY_LEN],
        },
    ];
    let parsed = cache::parse(&cache::serialize(&records));
    assert_eq!(parsed.env_count(), 3);
    assert_eq!(parsed.salvaged().len(), 2);
}

#[test]
fn environment_size_matches_bytes_written() {
    for env in [env_fragment(), compute_env(), geometry_env()] {
        let before = cache::serialize(&[]).len(); // header length
        let bytes = cache::serialize(&[Record {
            offset: 0,
            envs: vec![env.clone()],
            key: [0; KEY_LEN],
        }]);
        let written = bytes.len() - before;
        // Record = num_envs (4) + env + key (16)
        assert_eq!(
            written,
            4 + env.size() + KEY_LEN,
            "size() must match write()"
        );
    }
}

#[test]
fn damage_display_is_readable() {
    let d = Damage {
        offset: 42,
        reason: DamageKind::BadEnvCount { found: 300 },
    };
    let text = d.to_string();
    assert!(text.contains("42"), "{text}");
    assert!(text.contains("300"), "{text}");
}

#[test]
fn parsed_cache_is_debug_and_has_no_damage_when_clean() {
    let parsed: ParsedCache = cache::parse(&cache::serialize(&[record(1)]));
    assert!(format!("{parsed:?}").contains("Record"));
    assert!(parsed.damage.is_none());
}
