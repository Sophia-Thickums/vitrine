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
        let words = programs()
            .remove(name)
            .unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let jit_state =
            run_jit(&words, &mut memory, 0x1800, 200).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp_state = run_interp(&words, 0x1800);

        assert_eq!(
            jit_state.x,
            interp_state.x,
            "{name}: register file differs\n jit {:x?}\n int {:x?}",
            &jit_state.x[..8],
            &interp_state.x[..8]
        );
        assert_eq!(jit_state.sp, interp_state.sp, "{name}: sp differs");
        assert_eq!(
            jit_state.get_flags(),
            (
                interp_state.n,
                interp_state.z,
                interp_state.c,
                interp_state.v
            ),
            "{name}: flags differ"
        );
    }
}

#[test]
fn compiled_loops_and_branches_agree_with_the_interpreter() {
    for (name, limit) in [("loop_with_cbz", 200u64), ("compare_and_branch", 100)] {
        let words = programs()
            .remove(name)
            .unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let jit_state =
            run_jit(&words, &mut memory, 0x1800, limit).unwrap_or_else(|e| panic!("{name}: {e}"));
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

    assert_eq!(
        state.x[1..=3],
        interp_machine.x[1..=3],
        "registers after the pair"
    );
    assert_eq!(
        state.x[2], interp_machine.x[2],
        "post-index write-back must match"
    );
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
        let words = programs()
            .remove(name)
            .unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let state =
            run_jit(&words, &mut memory, 0x1800, 100).unwrap_or_else(|e| panic!("{name}: {e}"));
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
        let words = programs()
            .remove(name)
            .unwrap_or_else(|| panic!("fixture {name}"));
        let mut memory = vec![0u8; 0x2000];
        let state =
            run_jit(&words, &mut memory, 0x1800, 200).unwrap_or_else(|e| panic!("{name}: {e}"));
        let interp = run_interp(&words, 0x1800);
        assert_eq!(state.x, interp.x, "{name}: register file differs");
        assert_eq!(state.fp, interp.fp, "{name}: FP register file differs");
    }
}

// ── the data-abort guard: the thing that makes this JIT safe on a real guest ─────────────────────

#[test]
fn a_load_past_the_end_of_the_window_aborts_instead_of_reading_host_memory() {
    // ldr x0, [x1]  ; ret        — x1 is set to the last 4 bytes of the window, so the 8-byte load
    // straddles the end. Before the guard existed this read host memory past the allocation and
    // returned whatever was there.
    let program: &'static [u32] = &[0xF940_0020, 0xD65F_03C0];
    let mut memory = vec![0u8; 64];

    let mut state = JitState::new(0, 0);
    state.x[1] = 60; // 60..68 in a 64-byte window
    jit::run(program, &mut memory, &mut state, 8).expect("driver returns rather than dying");

    assert_eq!(
        state.last_exit,
        jit::Exit::DataAbort as u64,
        "last exit is a data abort"
    );
    assert_eq!(
        state.pc, 0,
        "pc points at the faulting instruction, so it can be retried"
    );
    let base = memory.as_ptr() as u64;
    assert_eq!(
        state.fault_addr,
        base + 60,
        "the faulting host address is recorded"
    );
}

#[test]
fn a_load_entirely_inside_the_window_still_runs() {
    // The same program with the load in bounds must not abort: a guard that refuses everything is
    // not a guard.
    let program: &'static [u32] = &[0xF940_0020, 0xD65F_03C0];
    let mut memory = vec![0u8; 64];
    let mut state = JitState::new(0, 0);
    state.x[1] = 0;
    state.x[0] = 0;
    jit::run(program, &mut memory, &mut state, 8).expect("runs");

    assert_ne!(
        state.last_exit,
        jit::Exit::DataAbort as u64,
        "in-bounds access must not abort"
    );
}

#[test]
fn a_pair_access_straddling_the_end_aborts_on_the_second_half() {
    // ldp x0, x1, [x2]  ; ret      (encoding from llvm-mc, not from memory)
    // A pair reads 8+8 bytes at [x2] and [x2+8]. With x2 = 56 in a 64-byte window, the FIRST half
    // lands and the SECOND does not — so the guard has to cover the whole span, not one element.
    // This is the arm a reviewer questioned: it settles by measurement whether size*2 is the right
    // span, rather than by either of us arguing.
    let program: &'static [u32] = &[0xA940_0440, 0xD65F_03C0];
    let mut memory = vec![0u8; 64];

    let mut state = JitState::new(0, 0);
    state.x[2] = 56;
    jit::run(program, &mut memory, &mut state, 8).expect("driver returns");

    assert_eq!(
        state.last_exit,
        jit::Exit::DataAbort as u64,
        "the second half of the pair is past the window, so the pair must abort"
    );
    assert_eq!(
        state.fault_addr,
        memory.as_ptr() as u64 + 56,
        "the faulting address is the base of the pair"
    );

    // And the same pair fully in bounds still runs.
    let mut state2 = JitState::new(0, 0);
    state2.x[2] = 40;
    jit::run(program, &mut memory, &mut state2, 8).expect("driver returns");
    assert_ne!(
        state2.last_exit,
        jit::Exit::DataAbort as u64,
        "in-bounds pair must not abort"
    );
}

// ── the local model's audit, turned into evidence rather than opinion ────────────────────────────

#[test]
fn a_wrapping_access_address_is_caught_by_the_below_window_check() {
    // Claim from a local-model review: `rcx = rax + size-1` can wrap when rax is near u64::MAX and
    // bypass the upper-bound test. The reasoning omitted the FIRST compare, which is a below-window
    // check — and a wrapped address lands BELOW the base, where `jb` catches it. Proved, not argued:
    // mem_ptr is set 4 bytes below the top of the address space, so base + 8 wraps to 3.
    let words: &'static [u32] = &[0xF940_0020, 0xD65F_03C0]; // ldr x0, [x1] ; ret
    let mut state = JitState::new(u64::MAX - 4, 64);
    state.pc = 0;
    state.x[1] = 8;

    let mut jit = jit::AddressJit::new(|a: u64| words.get((a / 4) as usize).copied(), state);
    jit.run(4).expect("driver returns rather than dying");

    assert_eq!(
        jit.state().last_exit,
        jit::Exit::DataAbort as u64,
        "a wrapped address must abort; the below-window compare is what catches it"
    );
    assert_eq!(
        jit.state().pc,
        0,
        "and pc stays on the faulting instruction"
    );
}

#[test]
fn the_last_byte_of_the_window_is_in_bounds_and_one_past_it_is_not() {
    // The second local-model claim was that `jae` against `mem_ptr + mem_len` is off by one. It is a
    // half-open window: the last readable byte is `mem_len-1`, and a load whose LAST byte is at
    // `mem_len-1` must succeed while one that reaches `mem_len` must abort. Both arms here.
    let words: &'static [u32] = &[0xF940_0020, 0xD65F_03C0]; // ldr x0, [x1] ; ret

    // ★ mem_ptr must be a REAL mapping: the guard bounds-checks the window, it does not verify that
    // the window is host memory. A fabricated base makes the in-bounds arm dereference whatever the
    // number points at — this test segfaulted with mem_ptr = 0, which is exactly what a caller of the
    // C ABI would do if it passed a stale pointer.
    let window = [0u8; 64];
    let base = window.as_ptr() as u64;
    let mut ok_state = JitState::new(base, 64);
    ok_state.pc = 0;
    ok_state.x[1] = 56; // bytes 56..63 -> last byte 63, inside a 64-byte window
    let mut jit = jit::AddressJit::new(|a: u64| words.get((a / 4) as usize).copied(), ok_state);
    jit.run(4).expect("runs");
    assert_ne!(
        jit.state().last_exit,
        jit::Exit::DataAbort as u64,
        "last byte inside must load"
    );

    let mut bad_state = JitState::new(base, 64);
    bad_state.pc = 0;
    bad_state.x[1] = 57; // bytes 57..64 -> last byte 64, one past the window
    let mut jit2 = jit::AddressJit::new(|a: u64| words.get((a / 4) as usize).copied(), bad_state);
    jit2.run(4).expect("returns");
    assert_eq!(
        jit2.state().last_exit,
        jit::Exit::DataAbort as u64,
        "one byte past must abort"
    );
}

#[test]
fn the_compiler_refuses_an_exclusive_instead_of_emitting_wrong_code() {
    // LL/SC is decoded and the interpreter executes it; the JIT does not emit it. The contract is
    // that it REFUSES — a block containing one must return Err so the driver stops visibly, never a
    // compiled block that silently skips the lock.
    let words: &'static [u32] = &[0xC85F_7C20, 0xD65F_03C0]; // ldxr x0, [x1] ; ret
    let program = block::scan(words);
    let blk = program.block_at(0).expect("block at 0");
    match jit::compile(blk) {
        Err(vitrine::a64::Instr::LoadExclusive { .. }) => {}
        Err(other) => panic!("expected the exclusive to be the refusal, got {other:?}"),
        Ok(_) => panic!("the compiler must not emit code for an exclusive it cannot honour"),
    }
}

// ── gigawatt's split of the tag arm (2026-11-01): three questions, not one ───────────────────────

#[test]
fn a_one_byte_access_is_bounded_by_the_same_guard_the_wide_ones_use() {
    // The question the earlier arms left open: "the guard covers the last byte" and "the guard is not
    // consulted for a one-byte access" are different claims, and a reader of `mem_guard` cannot tell
    // which one was built, because `size > 1` only skips the `+ size - 1`, it does not skip the call.
    // ldrb w0, [x1] ; ret
    let words: &'static [u32] = &[0x3940_0020, 0xD65F_03C0];
    let window = [0u8; 64];
    let base = window.as_ptr() as u64;

    let mut ok = JitState::new(base, 64);
    ok.pc = 0;
    ok.x[1] = 63;
    let mut jit = jit::AddressJit::new(|a: u64| words.get((a / 4) as usize).copied(), ok);
    jit.run(4).expect("runs");
    assert_ne!(
        jit.state().last_exit,
        jit::Exit::DataAbort as u64,
        "the last byte of the window is readable by a size-1 access"
    );

    let mut bad = JitState::new(base, 64);
    bad.pc = 0;
    bad.x[1] = 64;
    let mut jit2 = jit::AddressJit::new(|a: u64| words.get((a / 4) as usize).copied(), bad);
    jit2.run(4).expect("returns");
    assert_eq!(
        jit2.state().last_exit,
        jit::Exit::DataAbort as u64,
        "one byte past the window aborts for size 1 as well, so the guard IS consulted"
    );
}

#[test]
fn a_faulting_pair_leaves_both_destination_registers_untouched() {
    // gigawatt's third assertion: an ldp writes two registers, so an implementation that faults
    // correctly and corrupts the guest on the way passes `fault_addr == base + 56`. Here the guard
    // runs BEFORE either load and before any writeback, so nothing should be committed — this arm is
    // what turns "by construction" into evidence.
    // ldp x0, x1, [x2] ; ret
    let program: &'static [u32] = &[0xA940_0440, 0xD65F_03C0];
    let mut memory = vec![0u8; 64];
    let mut state = JitState::new(0, 0);
    state.x[2] = 56;
    state.x[0] = 0xDEAD_BEEF_0000_0000;
    state.x[1] = 0xFEED_FACE_0000_0001;
    jit::run(program, &mut memory, &mut state, 8).expect("driver returns");

    assert_eq!(
        state.last_exit,
        jit::Exit::DataAbort as u64,
        "the pair aborts"
    );
    assert_eq!(
        state.x[0], 0xDEAD_BEEF_0000_0000,
        "the first destination must not be written by a faulting pair"
    );
    assert_eq!(
        state.x[1], 0xFEED_FACE_0000_0001,
        "nor the second — the interesting failure is a correct abort that corrupts the guest"
    );
}

#[test]
fn a_low_bit_tagged_base_is_bounds_checked_as_given_not_stripped() {
    // gigawatt's mid-window case, recorded as a DECISION rather than left implicit: this guard does
    // not mask low bits. A tagged base inside the window therefore RUNS, at the tagged address, and
    // reports that same address in fault_addr when the tag pushes the span out. The two readings —
    // strip the tag, or refuse a tagged entry — are both defensible, and this arm is the record of
    // which one exists here: neither. It is a decision to make deliberately, not to inherit.
    // ldp x0, x1, [x2] ; ret
    let program: &'static [u32] = &[0xA940_0440, 0xD65F_03C0];

    // Mid-window, tag 0b10 set: base 32 | 2 = 34. The span [34, 50) is inside a 64-byte window, so
    // the guard passes and the access happens at 34 — shifted, not refused.
    let mut memory = vec![0u8; 64];
    for (i, b) in memory.iter_mut().enumerate() {
        *b = i as u8;
    }
    let mut state = JitState::new(0, 0);
    state.x[2] = 34;
    jit::run(program, &mut memory, &mut state, 8).expect("driver returns");
    assert_ne!(
        state.last_exit,
        jit::Exit::DataAbort as u64,
        "a tagged base that stays inside the window runs: the tag is not a fault here"
    );
    assert_eq!(
        state.x[0],
        u64::from_le_bytes([34, 35, 36, 37, 38, 39, 40, 41]),
        "and it reads at the tagged address, so the guest's low bits were not masked"
    );

    // End-of-window, tag straddles: base 62 | 3 = 63, span [63, 79) crosses the end. The abort must
    // report the tagged address, because that is the address the guest asked for.
    let mut memory2 = vec![0u8; 64];
    let mut st2 = JitState::new(0, 0);
    st2.x[2] = 63;
    jit::run(program, &mut memory2, &mut st2, 8).expect("driver returns");
    assert_eq!(
        st2.last_exit,
        jit::Exit::DataAbort as u64,
        "straddling span aborts"
    );
    assert_eq!(
        st2.fault_addr,
        memory2.as_ptr() as u64 + 63,
        "fault_addr is the tagged base, not a stripped one"
    );
}
