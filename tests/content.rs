//! Content selection tests: the behaviour that replaced "whichever file the directory walk
//! reached last".

use vitrine::content::{self, Entry, Rejected, Selection, TitleType, Warning, base_title_id};

const BASE: u64 = 0x0100_0000_0001_0000;
const PATCH: u64 = BASE + 0x800;
const DLC0: u64 = BASE + 0x1000;
const DLC1: u64 = BASE + 0x1001; // DLC index lives in the low bits of the +0x1000 block
const OTHER_BASE: u64 = 0x0100_0000_0002_0000;

fn app(version: u32) -> Entry {
    Entry { title_id: BASE, version, title_type: TitleType::Application, source: 1 }
}

fn patch(version: u32, source: u64) -> Entry {
    Entry { title_id: PATCH, version, title_type: TitleType::Patch, source }
}

fn dlc(title_id: u64, version: u32) -> Entry {
    Entry { title_id, version, title_type: TitleType::AddOnContent, source: 9 }
}

// -- title id arithmetic ----------------------------------------------------

#[test]
fn base_title_id_recovers_the_application_for_each_kind() {
    assert_eq!(base_title_id(BASE, TitleType::Application), BASE);
    assert_eq!(base_title_id(PATCH, TitleType::Patch), BASE);
    assert_eq!(base_title_id(DLC0, TitleType::AddOnContent), BASE);
    assert_eq!(base_title_id(DLC1, TitleType::AddOnContent), BASE);
    assert_eq!(
        base_title_id(0x0100_0000_0002_0800, TitleType::Patch),
        OTHER_BASE
    );
}

#[test]
fn aoc_index_reads_the_low_bits() {
    assert_eq!(content::aoc_index(DLC0, TitleType::AddOnContent), Some(0));
    assert_eq!(content::aoc_index(DLC1, TitleType::AddOnContent), Some(1));
    assert_eq!(content::aoc_index(PATCH, TitleType::Patch), None);
}

#[test]
fn entry_computes_its_own_base() {
    assert_eq!(patch(1, 1).base_title_id(), BASE);
    assert_eq!(dlc(DLC0, 0).base_title_id(), BASE);
}

// -- the core fix -----------------------------------------------------------

#[test]
fn highest_patch_version_wins_regardless_of_input_order() {
    let entries = [app(0), patch(0x10000, 1), patch(0x30000, 2), patch(0x20000, 3)];
    let forward = content::select(entries.iter().cloned());
    assert_eq!(
        forward.patch(BASE).map(|e| e.version),
        Some(0x30000),
        "the newest update must be selected"
    );
    assert_eq!(forward.rejected.len(), 2);
    for (_, reason) in &forward.rejected {
        assert_eq!(
            reason,
            &Rejected::Superseded { chosen_version: 0x30000 },
            "losers must name the version that beat them"
        );
    }

    // Reverse the input: the same winner must come out.
    let mut reversed: Vec<Entry> = entries.to_vec();
    reversed.reverse();
    let backward = content::select(reversed);
    assert_eq!(backward.patch(BASE).map(|e| e.version), Some(0x30000));
}

#[test]
fn selection_is_permutation_invariant() {
    // Every permutation of the same four entries must produce byte-identical choices.
    let entries = [app(0), patch(0x10000, 1), patch(0x30000, 2), patch(0x20000, 3)];
    let baseline = content::select(entries.iter().cloned());
    for perm in permutations(&entries) {
        let s = content::select(perm);
        assert_eq!(
            s.chosen.iter().map(|(k, v)| (*k, v.version)).collect::<Vec<_>>(),
            baseline.chosen.iter().map(|(k, v)| (*k, v.version)).collect::<Vec<_>>(),
            "selection must not depend on input order"
        );
    }
}

fn permutations(items: &[Entry]) -> Vec<Vec<Entry>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

#[test]
fn identical_content_installed_twice_is_reported_as_a_duplicate() {
    let entries = [
        Entry { title_id: PATCH, version: 0x10000, title_type: TitleType::Patch, source: 5 },
        Entry { title_id: PATCH, version: 0x10000, title_type: TitleType::Patch, source: 5 },
    ];
    let s = content::select(entries);
    assert_eq!(s.chosen.len(), 1);
    assert_eq!(s.rejected.len(), 1);
    assert_eq!(s.rejected[0].1, Rejected::Duplicate);
}

#[test]
fn same_version_different_file_is_resolved_deterministically_and_reported() {
    let entries = [
        Entry { title_id: PATCH, version: 7, title_type: TitleType::Patch, source: 20 },
        Entry { title_id: PATCH, version: 7, title_type: TitleType::Patch, source: 10 },
    ];
    let s = content::select(entries);
    // Lowest source wins the tie, so the outcome never depends on walk order.
    assert_eq!(s.chosen[&PATCH].source, 10);
    assert_eq!(s.rejected.len(), 1);
}

// -- slots do not compete ---------------------------------------------------

#[test]
fn patch_does_not_replace_the_application() {
    let s = content::select([app(0), patch(0x10000, 1)]);
    assert_eq!(s.chosen.len(), 2, "application and patch are different slots");
    assert!(s.application(BASE).is_some());
    assert!(s.patch(BASE).is_some());
    assert!(s.rejected.is_empty());
}

#[test]
fn two_games_do_not_interfere() {
    let s = content::select([
        app(0),
        patch(0x10000, 1),
        Entry {
            title_id: OTHER_BASE,
            version: 0,
            title_type: TitleType::Application,
            source: 2,
        },
        Entry {
            title_id: OTHER_BASE + 0x800,
            version: 0x20000,
            title_type: TitleType::Patch,
            source: 3,
        },
    ]);
    assert_eq!(
        s.patch(BASE).map(|e| e.version),
        Some(0x10000)
    );
    assert_eq!(
        s.patch(OTHER_BASE).map(|e| e.version),
        Some(0x20000)
    );
}

// -- DLC -------------------------------------------------------------------

#[test]
fn all_dlc_for_a_base_are_returned_ordered_by_title_id() {
    let s = content::select([app(0), dlc(DLC1, 0), dlc(DLC0, 0)]);
    let ids: Vec<u64> = s.dlc(BASE).iter().map(|e| e.title_id).collect();
    assert_eq!(ids, vec![DLC0, DLC1], "ordered by title id");
}

#[test]
fn dlc_for_another_game_is_not_returned() {
    let s = content::select([
        app(0),
        dlc(DLC0, 0),
        dlc(OTHER_BASE + 0x1000, 0),
    ]);
    assert_eq!(s.dlc(BASE).len(), 1);
    assert_eq!(s.dlc(OTHER_BASE).len(), 1);
}

#[test]
fn higher_dlc_version_supersedes_lower() {
    let s = content::select([app(0), dlc(DLC0, 0), dlc(DLC0, 0x1000)]);
    assert_eq!(s.dlc(BASE)[0].version, 0x1000);
    assert_eq!(s.rejected.len(), 1);
}

// -- warnings --------------------------------------------------------------

#[test]
fn patch_without_its_application_warns() {
    let s = content::select([patch(0x10000, 1)]);
    assert!(s.application(BASE).is_none());
    assert!(s.patch(BASE).is_some(), "the patch is still reported");
    assert_eq!(
        s.warnings,
        vec![Warning::OrphanPatch { title_id: PATCH, version: 0x10000 }]
    );
}

#[test]
fn dlc_without_its_application_warns() {
    let s = content::select([dlc(DLC0, 0)]);
    assert_eq!(
        s.warnings,
        vec![Warning::OrphanDlc { title_id: DLC0, version: 0 }]
    );
}

#[test]
fn a_complete_install_produces_no_warnings() {
    let s = content::select([app(0), patch(0x10000, 1), dlc(DLC0, 0)]);
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
}

// -- accounting ------------------------------------------------------------

#[test]
fn every_input_entry_is_accounted_for() {
    let entries = vec![
        app(0),
        patch(0x10000, 1),
        patch(0x20000, 2),
        patch(0x30000, 3),
        dlc(DLC0, 0),
        dlc(DLC1, 0),
        dlc(DLC1, 5),
    ];
    let n = entries.len();
    let s = content::select(entries);
    assert_eq!(
        s.total_inputs(),
        n,
        "no entry may vanish: {} chosen + {} rejected != {n}",
        s.chosen.len(),
        s.rejected.len()
    );
}

#[test]
fn empty_input_yields_empty_selection() {
    let s = content::select(Vec::<Entry>::new());
    assert!(s.chosen.is_empty());
    assert!(s.rejected.is_empty());
    assert!(s.warnings.is_empty());
    assert_eq!(s.total_inputs(), 0);
}

#[test]
fn display_summarises_the_selection() {
    let s: Selection = content::select([app(0), patch(0x10000, 1), patch(0x20000, 2)]);
    let text = s.to_string();
    assert!(text.contains("1 application"), "{text}");
    assert!(text.contains("1 patch"), "{text}");
    assert!(text.contains("1 superseded"), "{text}");
}

#[test]
fn title_type_discriminants_match_the_cnmt_header() {
    assert_eq!(TitleType::from_u8(0x80), Some(TitleType::Application));
    assert_eq!(TitleType::from_u8(0x81), Some(TitleType::Patch));
    assert_eq!(TitleType::from_u8(0x82), Some(TitleType::AddOnContent));
    assert_eq!(TitleType::from_u8(0x83), Some(TitleType::Delta));
    assert_eq!(TitleType::from_u8(0x00), None);
    assert_eq!(TitleType::from_u8(0xFF), None);
}
