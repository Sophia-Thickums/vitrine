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
    assert_eq!(b0.exit, Exit::Fallthrough, "a call returns to the next instruction");
    assert!(b0.is_compilable());
    assert!(p.block_at(8).is_some(), "the instruction after the bl starts a block");
}

#[test]
fn an_undecodable_instruction_marks_its_block_uncompilable() {
    // A vector load, which this decoder deliberately does not implement.
    let words = [0x3DC0_0000, 0xD65F_03C0];
    let p = block::scan(&words);
    let b = p.block_at(0).expect("block at 0");
    assert!(b.has_unknown);
    assert!(!b.is_compilable(), "a block with an undecoded word must not be compiled");
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
    assert_eq!(p.block_at(0).expect("block at 0").exit, Exit::Direct { target: -4 });
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
    assert_eq!(p.instruction_count(), 4, "no word may be lost or duplicated");
}

#[test]
fn an_empty_region_produces_one_empty_block() {
    let p = block::scan(&[]);
    // No words, so nothing to compile; the map may hold an empty block at 0 and that is fine, but
    // no instruction may be invented.
    assert_eq!(p.instruction_count(), 0);
}
