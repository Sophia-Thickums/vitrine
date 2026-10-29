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
use vitrine::jit::{self, Exit, JitState};

const PAGE: usize = 4096;

/// An executable mapping holding one block.
struct Exec {
    ptr: *mut u8,
    len: usize,
}

impl Exec {
    fn new(code: &[u8]) -> Self {
        let len = code.len().div_ceil(PAGE) * PAGE;
        // SAFETY: an anonymous executable mapping with the sizes given; it is unmapped in Drop.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");
        // SAFETY: the mapping is `len` bytes and writable, and `code` is shorter.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
        }
        Self {
            ptr: ptr.cast::<u8>(),
            len,
        }
    }

    fn entry(&self) -> extern "C" fn(*mut JitState) -> u64 {
        // SAFETY: the caller asserts the mapping holds one compiled function with this signature.
        unsafe { std::mem::transmute::<*mut u8, extern "C" fn(*mut JitState) -> u64>(self.ptr) }
    }
}

impl Drop for Exec {
    fn drop(&mut self) {
        // SAFETY: the pointer and length are the ones returned by mmap in `new`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

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

/// Run `words` through the JIT, block by block, until the exit syscall or a limit.
///
/// This is a driver, not an optimisation: it compiles each block the interpreter would enter and
/// calls it. That is enough to prove the generated code agrees with the interpreter.
fn run_jit(words: &[u32], memory: &mut [u8], sp: u64, limit: u64) -> Result<JitState, String> {
    let mut state = JitState::new(memory.as_mut_ptr() as u64, memory.len() as u64);
    state.sp = sp;
    state.pc = 0;

    for _ in 0..limit {
        let pc = state.pc as usize;
        if pc >= words.len() * 4 {
            // Fell off the end of the program — the interpreter stops the same way with
            // PcOutOfRange, so treat it as a normal finish and compare what was reached.
            return Ok(state);
        }
        // Compile the block that starts at the current pc, by scanning from there.
        let program = block::scan(&words[pc / 4..]);
        let block = program
            .block_at(0)
            .ok_or_else(|| format!("no block at {pc:#x}"))?;
        // Offsets inside the sub-scan are relative; shift them to absolute.
        let mut absolute = block.clone();
        absolute.start += pc as i64;
        absolute.end += pc as i64;
        for ins in &mut absolute.instructions {
            ins.0 += pc as i64;
        }

        let code = jit::compile(&absolute).map_err(|i| format!("unsupported: {i:?}"))?;
        let exec = Exec::new(&code);
        let f = exec.entry();
        // SAFETY: `f` was compiled with this signature and `state` outlives the call.
        let exit = f(&mut state);
        match exit {
            x if x == Exit::Next as u64 => {}
            x if x == Exit::Jumped as u64 => {}
            x if x == Exit::SupervisorCall as u64 => return Ok(state),
            x if x == Exit::Unsupported as u64 => return Err("unsupported instruction".into()),
            other => return Err(format!("bad exit code {other}")),
        }
        if state.pc == pc as u64 {
            return Err("block did not advance".into());
        }
    }
    Err("step limit".into())
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
    let mut interp_machine = Machine::new(0, 0x2000);
    interp_machine.load_code(0, &words);
    interp_machine.sp = 0x1800;
    interp_machine.x[1] = 0xDEAD_BEEF;
    interp_machine.x[2] = 0x800;

    // The JIT driver starts from a zeroed state, so seed the same values before running.
    let mut state = JitState::new(jit_memory.as_mut_ptr() as u64, jit_memory.len() as u64);
    state.sp = 0x1800;
    state.x[1] = 0xDEAD_BEEF;
    state.x[2] = 0x800;

    // Compile the single block and run it.
    let program = block::scan(&words);
    let b = program.block_at(0).expect("block");
    let code = jit::compile(b).expect("compile");
    let exec = Exec::new(&code);
    let f = exec.entry();
    // SAFETY: compiled with this signature; `state` outlives the call.
    let _ = f(&mut state);

    for _ in 0..2 {
        interp_machine.step().expect("interpreter step");
    }
    assert_eq!(state.x[1..=3], interp_machine.x[1..=3], "registers after the pair");
    assert_eq!(
        state.x[2], interp_machine.x[2],
        "post-index write-back must match"
    );
    // Compare the touched memory.
    let offset = 0x808usize;
    assert_eq!(
        &jit_memory[offset..offset + 8],
        &interp_machine.memory[offset..offset + 8],
        "stored bytes must match"
    );
}

#[test]
fn an_unsupported_instruction_is_refused_not_miscompiled() {
    // `ret` is not emitted by this translator yet: it must come back as Err, never as a silent nop.
    let words = [0xD65F_03C0u32];
    let program = block::scan(&words);
    let b = program.block_at(0).expect("block");
    match jit::compile(b) {
        Err(_) => {}
        Ok(_) => panic!("`ret` must not compile silently"),
    }
}
