//! Compiled blocks against the interpreter, on the same programs.
//!
//! The chain of trust is: `qemu-aarch64` pins the interpreter (`examples/diff_qemu`), and this file
//! pins the compiler to the interpreter. If all three agree, the compiler is right about the
//! instructions qemu exercised.
//!
//! Generated code is executed for real — mapped with `mmap(PROT_EXEC)` and called through a function
//! pointer. A JIT that is merely disassembled has not been tested.

use std::collections::BTreeMap;

use vitrine::block;
use vitrine::interp::Machine;
use vitrine::jit::{self, JitState};

fn programs() -> BTreeMap<String, Vec<u32>> {
    let raw = include_str!("fixtures/programs.tsv");
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut f = line.split('\t');
        let name = f.next().expect("name").to_string();
        let words = f
            .next()
            .expect("words")
            .split_whitespace()
            .map(|w| u32::from_str_radix(w.trim_start_matches("0x"), 16).expect("hex"))
            .collect();
        out.insert(name, words);
    }
    out
}

/// Run through the compiled path, using the library's driver.
fn run_jit(words: &[u32], memory: &mut [u8], sp: u64, limit: u64) -> Result<JitState, String> {
    let mut state = JitState::new(0, 0);
    state.sp = sp;
    jit::run(words, memory, &mut state, limit).map(|_| state)
}

/// The interpreter's final state for the same program.
fn run_interp(words: &[u32], sp: u64) -> Machine {
    let mut m = Machine::new(0, 0x2000);
    m.load_code(0, words);
    m.sp = sp;
    let _ = m.run(10_000);
    m
}

#[test]
fn compiled_arithmetic_agrees_with_the_interpreter() {
    for name in [
        "cmp_zero_vs_one",
        "cmp_five_vs_one",
        "add_one_to_x2",
        "adds_one_to_x2",
        "and_immediate",
        "mul_and_add",
        "csel_eq",
        "udiv_by_zero",
    ] {
        let words = programs().remove(name).unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let jit_state = run_jit(&words, &mut memory, 0x1800, 200).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp_state = run_interp(&words, 0x1800);

        assert_eq!(
            jit_state.x, interp_state.x,
            "{name}: register file differs\n jit {:x?}\n int {:x?}",
            &jit_state.x[..8],
            &interp_state.x[..8]
        );
        assert_eq!(jit_state.sp, interp_state.sp, "{name}: sp differs");
        assert_eq!(
            jit_state.get_flags(),
            (interp_state.n, interp_state.z, interp_state.c, interp_state.v),
            "{name}: flags differ"
        );
    }
}

#[test]
fn compiled_loops_and_branches_agree_with_the_interpreter() {
    for (name, limit) in [("loop_with_cbz", 200u64), ("compare_and_branch", 100)] {
        let words = programs().remove(name).unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let jit_state = run_jit(&words, &mut memory, 0x1800, limit).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp_state = run_interp(&words, 0x1800);
        assert_eq!(jit_state.x, interp_state.x, "{name}: register file differs");
    }
}

#[test]
fn compiled_memory_access_agrees_with_the_interpreter() {
    let words = programs().remove("pre_post_index").expect("pre_post_index");
    let mut jit_memory = vec![0u8; 0x2000];
    let mut state = JitState::new(0, 0);
    state.sp = 0x1800;
    state.x[1] = 0xDEAD_BEEF;
    state.x[2] = 0x800;
    jit::run(&words, &mut jit_memory, &mut state, 1).expect("jit run");

    let mut interp_machine = Machine::new(0, 0x2000);
    interp_machine.load_code(0, &words);
    interp_machine.sp = 0x1800;
    interp_machine.x[1] = 0xDEAD_BEEF;
    interp_machine.x[2] = 0x800;
    for _ in 0..2 {
        interp_machine.step().expect("interpreter step");
    }

    assert_eq!(state.x[1..=3], interp_machine.x[1..=3], "registers after the pair");
    assert_eq!(state.x[2], interp_machine.x[2], "post-index write-back must match");
    let offset = 0x808usize;
    assert_eq!(
        &jit_memory[offset..offset + 8],
        &interp_machine.memory[offset..offset + 8],
        "stored bytes must match"
    );
}

#[test]
fn compiled_tbz_and_adr_agree_with_the_interpreter() {
    for name in ["tbz_branch", "adr_page"] {
        let words = programs().remove(name).unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let state = run_jit(&words, &mut memory, 0x1800, 100).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp = run_interp(&words, 0x1800);
        assert_eq!(state.x, interp.x, "{name}: register file differs");
    }
}

#[test]
fn an_unsupported_instruction_is_refused_not_miscompiled() {
    // An `MRS` has no code emitted for it (there are no system registers yet): it must come back
    // as Err, never as a silent no-op that leaves a stale value in the destination.
    let words = [0xD53B_D040u32]; // mrs x0, tpidr_el0
    let program = block::scan(&words);
    let b = program.block_at(0).expect("block");
    match jit::compile(b) {
        Err(_) => {}
        Ok(_) => panic!("`mrs` must not compile silently"),
    }
}

#[test]
fn compiled_stack_pairs_agree_with_the_interpreter() {
    let words = programs().remove("pair_stack").expect("pair_stack");
    let mut memory = vec![0u8; 0x2000];
    let state = run_jit(&words, &mut memory, 0x1800, 100).expect("jit run");
    let interp = run_interp(&words, 0x1800);
    assert_eq!(state.x, interp.x, "register file after stp/ldp");
    assert_eq!(state.sp, interp.sp, "sp must be restored");
}

#[test]
fn a_thousand_iteration_loop_agrees_with_the_interpreter() {
    let words = programs().remove("loop_1000").expect("loop_1000");
    let mut memory = vec![0u8; 0x2000];
    let state = run_jit(&words, &mut memory, 0x1800, 5000).expect("jit run");
    let interp = run_interp(&words, 0x1800);
    assert_eq!(state.x, interp.x, "1000 iterations of add 7");
}

#[test]
fn compiled_floating_point_agrees_with_the_interpreter() {
    for name in ["fp_add", "fp_mul_div"] {
        let words = programs().remove(name).unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let state = run_jit(&words, &mut memory, 0x1800, 200).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp = run_interp(&words, 0x1800);
        assert_eq!(state.x, interp.x, "{name}: register file differs");
        assert_eq!(state.fp, interp.fp, "{name}: FP register file differs");
    }
}
