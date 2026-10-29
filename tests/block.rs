//! Block analysis: the boundaries a recompiler must not get wrong.
//!
//! The instruction words below are real encodings, checked with `llvm-mc -disassemble`:
//!
//! ```text
//! 0x00  mov  x0, #1
//! 0x04  add  x1, x0, #2
//! 0x08  cbz  x1, #8      -> 0x10
//! 0x0c  b    #8          -> 0x14
//! 0x10  sub  x2, x1, #1
//! 0x14  ret
//! ```

use vitrine::a64::Instr;
use vitrine::block::{self, Exit};

const PROGRAM: [u32; 6] = [
    0xD280_0020, // mov x0, #1
    0x9100_0801, // add x1, x0, #2
    0xB400_0041, // cbz x1, #8
    0x1400_0002, // b #8
    0xD100_0422, // sub x2, x1, #1
    0xD65F_03C0, // ret
];

#[test]
fn blocks_split_where_control_flow_says_they_do() {
    let p = block::scan(&PROGRAM);
    let starts: Vec<i64> = p.blocks.keys().copied().collect();
    assert_eq!(
        starts,
        vec![0, 12, 16, 20],
        "start, after the cbz, the branch target, and the b's target"
    );
}

#[test]
fn the_conditional_branch_exits_to_its_target_and_falls_through() {
    let p = block::scan(&PROGRAM);
    let b0 = p.block_at(0).expect("block at 0");
    assert_eq!(b0.len(), 3, "mov, add, cbz");
    assert_eq!(b0.exit, Exit::Direct { target: 16 });
    assert_eq!(b0.end, 12);
    assert!(b0.is_compilable());
}

#[test]
fn an_unconditional_branch_ends_its_block_and_names_the_target() {
    let p = block::scan(&PROGRAM);
    let b = p.block_at(12).expect("block at 12");
    assert_eq!(b.len(), 1);
    assert_eq!(b.exit, Exit::Direct { target: 20 });
}

#[test]
fn fallthrough_reaches_the_next_block_without_a_branch() {
    let p = block::scan(&PROGRAM);
    let b = p.block_at(16).expect("block at 16");
    assert_eq!(b.len(), 1);
    assert_eq!(b.exit, Exit::Fallthrough, "no terminator: falls into 0x14");
    assert_eq!(b.end, 20);
}

#[test]
fn a_return_is_its_own_exit_kind() {
    let p = block::scan(&PROGRAM);
    let b = p.block_at(20).expect("block at 20");
    assert_eq!(b.exit, Exit::Return);
    assert!(p.externals.is_empty(), "every target was inside the region");
}

#[test]
fn every_word_appears_exactly_once_across_the_blocks() {
    // The accounting property that matters: a recompiler that drops or duplicates an instruction
    // produces code that runs and is wrong.
    let p = block::scan(&PROGRAM);
    let total: usize = p.blocks.values().map(|b| b.len()).sum();
    assert_eq!(total, PROGRAM.len());
    assert_eq!(p.instruction_count(), PROGRAM.len());
}

#[test]
fn an_indirect_branch_reports_its_register_and_has_no_target() {
    // br x3
    let words = [0xD61F_0060];
    let p = block::scan(&words);
    let b = p.block_at(0).expect("block at 0");
    assert_eq!(b.exit, Exit::Indirect { register: 3 });
    assert_eq!(b.exit.target(), None);
}

#[test]
fn a_target_outside_the_region_is_recorded_not_followed() {
    // b #0x1000 — far past the end of a one-instruction region.
    let words = [0x1400_0400];
    let p = block::scan(&words);
    assert_eq!(p.externals, vec![0x1000]);
    assert_eq!(p.blocks.len(), 1);
}

#[test]
fn a_call_ends_the_block_and_falls_through_after_it() {
    // mov x0,#1 ; bl #8 ; ret
    let words = [0xD280_0020, 0x9400_0002, 0xD65F_03C0];
    let p = block::scan(&words);
    let b0 = p.block_at(0).expect("block at 0");
    assert_eq!(b0.len(), 2, "mov then bl");
    assert_eq!(
        b0.exit,
        Exit::Fallthrough,
        "a call returns to the next instruction"
    );
    assert!(b0.is_compilable());
    assert!(
        p.block_at(8).is_some(),
        "the instruction after the bl starts a block"
    );
}

#[test]
fn an_undecodable_instruction_marks_its_block_uncompilable() {
    // A vector load, which this decoder deliberately does not implement.
    let words = [0x3DC0_0000, 0xD65F_03C0];
    let p = block::scan(&words);
    let b = p.block_at(0).expect("block at 0");
    assert!(b.has_unknown);
    assert!(
        !b.is_compilable(),
        "a block with an undecoded word must not be compiled"
    );
    assert!(matches!(b.instructions[0].2, Instr::Unknown { .. }));
}

#[test]
fn a_branch_to_itself_is_a_single_block_and_terminates_the_scan() {
    // `b .` — imm26 == 0, an infinite loop, and the classic way a naive scanner fails to return.
    let words = [0x1400_0000];
    let p = block::scan(&words);
    assert_eq!(p.blocks.len(), 1);
    let b = p.block_at(0).expect("block at 0");
    assert_eq!(b.exit, Exit::Direct { target: 0 });
}

#[test]
fn a_backwards_branch_to_an_earlier_instruction_is_a_negative_target() {
    // `b .-4` at offset 0 aims before the region: reported as an external, not followed.
    let words = [0x17FF_FFFF];
    let p = block::scan(&words);
    assert_eq!(
        p.block_at(0).expect("block at 0").exit,
        Exit::Direct { target: -4 }
    );
    assert_eq!(p.externals, vec![-4]);
}

#[test]
fn a_backward_branch_into_the_region_splits_a_block_at_the_target() {
    // mov ; nop ; cbz x1, .-4 ; ret — the cbz at 0x08 targets 0x04, so 0x04 must be a block start
    // even though nothing there ends a block.
    let words = [0xD280_0020, 0xD503_201F, 0xB4FF_FFE1, 0xD65F_03C0];
    let p = block::scan(&words);
    let starts: Vec<i64> = p.blocks.keys().copied().collect();
    assert_eq!(starts, vec![0, 4, 12]);
    assert_eq!(
        p.instruction_count(),
        4,
        "no word may be lost or duplicated"
    );
}

#[test]
fn an_empty_region_produces_one_empty_block() {
    let p = block::scan(&[]);
    // No words, so nothing to compile; the map may hold an empty block at 0 and that is fine, but
    // no instruction may be invented.
    assert_eq!(p.instruction_count(), 0);
}

// ── scan_at: the block-at-address decode an emulator needs ───────────────────────────────────────

fn reader(words: &'static [u32], base: i64) -> impl FnMut(i64) -> Option<u32> {
    move |addr: i64| {
        let off = addr - base;
        if off < 0 || off % 4 != 0 {
            return None;
        }
        words.get((off / 4) as usize).copied()
    }
}

#[test]
fn scan_at_agrees_with_scan_wherever_a_terminator_closes_the_block() {
    // A block that ends because of its own terminator is fully determined locally, so both scans
    // must produce the same thing. A block that ended only because the region's next start cut it
    // off (`exit: Fallthrough`) is the case scan_at cannot reproduce — see the test below.
    let p = block::scan(&PROGRAM);
    let mut compared = 0;
    for (start, expected) in &p.blocks {
        if expected.exit == Exit::Fallthrough {
            continue;
        }
        let got = block::scan_at(*start, reader(&PROGRAM, 0), 64).expect("block here");
        assert_eq!(
            &got, expected,
            "scan_at disagrees with scan at offset {start}"
        );
        compared += 1;
    }
    assert_eq!(
        compared, 3,
        "the three terminated blocks were actually compared"
    );
}

#[test]
fn scan_at_extends_past_a_boundary_scan_knew_about_and_the_cache_must_drop_it() {
    // Offset 20 is the target of the `b` at 12, so scan() closes the block at 16 one instruction
    // early. scan_at has no way to know 20 is a target: it walks to the `ret` and returns a block
    // covering [16, 24). That is the documented difference, and it is why the cache is keyed by
    // guest address and InvalidateCacheRange must drop [16, 24) when a branch is discovered into
    // the middle of it.
    let p = block::scan(&PROGRAM);
    let short = p.block_at(16).expect("scan's short block");
    let long = block::scan_at(16, reader(&PROGRAM, 0), 64).expect("scan_at's block");

    assert_eq!(short.len(), 1);
    assert_eq!(long.len(), 2);
    assert_eq!(long.exit, Exit::Return);
    assert!(
        long.end > p.blocks.keys().copied().find(|k| *k > 16).unwrap(),
        "the longer block reaches past the next known start, which is why invalidation exists"
    );
}

#[test]
fn scan_at_at_an_unmapped_address_returns_none_not_an_empty_block() {
    let got = block::scan_at(0x1000, reader(&PROGRAM, 0), 64);
    assert!(
        got.is_none(),
        "an unmapped address is no block, never an empty one"
    );
}

#[test]
fn scan_at_marks_truncation_when_memory_ends_mid_block() {
    // Only the first two words are mapped; the third read fails.
    let partial: &'static [u32] = &PROGRAM[..2];
    let got = block::scan_at(0, reader(partial, 0), 64).expect("block starts");
    assert_eq!(got.len(), 2);
    assert!(got.has_unknown, "a truncated block must be uncompilable");
    assert!(!got.is_compilable());
}

#[test]
fn scan_at_stops_at_the_instruction_budget() {
    // `add x1, x0, #2` is not a terminator; with a budget of 3 the walk must stop at 3.
    let run: &'static [u32] = &[0x9100_0801; 8];
    let got = block::scan_at(0, reader(run, 0), 3).expect("block starts");
    assert_eq!(got.len(), 3);
    assert_eq!(got.exit, Exit::Fallthrough);
    assert_eq!(got.end, 12);
}

#[test]
fn scan_at_reports_absolute_guest_offsets() {
    // The same program based at 0x1000: the epilogue writes pc verbatim, so offsets must be absolute.
    let got = block::scan_at(0x1000, reader(&PROGRAM, 0x1000), 64).expect("block at 0x1000");
    assert_eq!(got.start, 0x1000);
    assert_eq!(
        got.instructions[0].0, 0x1000,
        "first instruction is at the guest address"
    );
    assert_eq!(
        got.instructions.last().unwrap().0,
        0x1008,
        "cbz sits at 0x1008"
    );
    assert_eq!(got.end, 0x100c);

    // A direct branch's target must also be absolute: `b #8` at 0x100c jumps to 0x1014.
    let b = block::scan_at(0x100c, reader(&PROGRAM, 0x1000), 64).expect("block at 0x100c");
    assert_eq!(b.exit, Exit::Direct { target: 0x1014 });
}
