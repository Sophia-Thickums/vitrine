//! Block compiler: AArch64 words in, x86-64 machine code out.
//!
//! # Shape
//!
//! A compiled block is one function with the System V signature
//!
//! ```text
//! extern "C" fn(state: *mut JitState) -> u64
//! ```
//!
//! returning an [`Exit`] code. Guest registers live in memory at the offsets in
//! [`crate::x86::state`], so a block never spills: it loads what it needs, computes in `rax`/`rcx`/
//! `rdx`/`rsi`, and stores back. `r15` holds the state pointer for the whole function.
//!
//! This is a **correctness-first** translator. No register allocation, no block chaining, no
//! superblocks. Speed comes after the semantics are proven, because a JIT that is subtly wrong is
//! worse than an interpreter that is slow.
//!
//! # Verification
//!
//! `tests/jit.rs` runs the same programs through the interpreter and through compiled code and
//! requires identical architectural state — and `examples/diff_qemu.rs` already pins the interpreter
//! to qemu. So the chain is qemu → interpreter → JIT.

use crate::a64::{AddressMode, Cond, FpOp, Instr, LogicalOp, MovOp, ShiftType};
use crate::block::Block;
use crate::x86::{Cc, Code, Reg, state};

/// Why a compiled block returned.
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Fell off the end; `pc` holds the next block.
    Next = 0,
    /// `pc` was written by the block (branch, return, indirect jump).
    Jumped = 1,
    /// A supervisor call was reached; `pc` points at it.
    SupervisorCall = 2,
    /// An instruction this translator does not emit.
    Unsupported = 3,
    /// A guest memory access left the window; `pc` points at the faulting instruction and
    /// [`JitState::fault_addr`] holds the address it tried to reach.
    ///
    /// Before this existed, generated code performed every access with no check at all — `mem_len`
    /// was written by the driver and read by nobody — so a guest address outside the window read or
    /// wrote host memory. For a benchmark that never happened; for an emulator it is the first
    /// unmapped page.
    DataAbort = 4,
}

/// The state block generated code operates on.
///
/// Field order and offsets are ABI: [`crate::x86::state`] names them, and `state_layout_is_stable`
/// checks them against `size_of`/`offset_of`.
#[repr(C)]
pub struct JitState {
    /// `x0..x30`.
    pub x: [u64; 31],
    /// Stack pointer.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// Packed flags: bit 3 N, 2 Z, 1 C, 0 V.
    pub flags: u8,
    _pad: [u8; 7],
    /// Host pointer to byte 0 of guest memory.
    pub mem_ptr: u64,
    /// Guest memory length, for bounds (checked by the driver, not by generated code).
    pub mem_len: u64,
    /// `V0..V31`, 16 bytes each.
    pub fp: [[u8; 16]; 32],
    /// Guest address of the faulting access, set only on [`Exit::DataAbort`].
    ///
    /// Appended at the end deliberately: every field above keeps the offset it had, because
    /// `state_layout_is_stable` pins those offsets and the generated code hard-codes them.
    pub fault_addr: u64,
    /// Why the driver last stopped, as an [`Exit`] discriminant. Written by the driver, not by
    /// generated code — a fault stub cannot write it without a second store on the hot path, and
    /// the driver knows the exit code the moment the block returns.
    pub last_exit: u64,
}

impl JitState {
    /// A fresh state with the given guest memory already mapped by the caller.
    pub fn new(mem_ptr: u64, mem_len: u64) -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            flags: 0,
            _pad: [0; 7],
            mem_ptr,
            mem_len,
            fp: [[0; 16]; 32],
            fault_addr: 0,
            last_exit: 0,
        }
    }

    /// Pack flags for generated code.
    pub fn set_flags(&mut self, n: bool, z: bool, c: bool, v: bool) {
        self.flags = u8::from(n) << 3 | u8::from(z) << 2 | u8::from(c) << 1 | u8::from(v);
    }

    /// Unpack flags.
    pub fn get_flags(&self) -> (bool, bool, bool, bool) {
        (
            self.flags & 0b1000 != 0,
            self.flags & 0b0100 != 0,
            self.flags & 0b0010 != 0,
            self.flags & 0b0001 != 0,
        )
    }
}

/// Compile one block. `block` must be compilable (no undecoded words).
///
/// Returns `Err` with the offending instruction when the translator has no code for it, so an
/// unhandled instruction is visible rather than silently skipped.
pub fn compile(block: &Block) -> Result<Vec<u8>, Instr> {
    let mut c = Code::new();

    // Prologue: preserve the callee-saved registers this function uses, then take the state
    // pointer into r15. r15 must be saved as well as restored: it is the caller's.
    c.push_reg(Reg::Rbx);
    c.push_reg(Reg::R15);
    c.mov_reg(Reg::R15, Reg::Rdi);

    for &(offset, _word, instr) in &block.instructions {
        emit(&mut c, instr, offset)?;
    }

    // Fallthrough epilogue: pc = end of block, return Next.
    epilogue(&mut c, block.end, Exit::Next);

    Ok(c.finish())
}


/// Labels for the data-abort stub guarding the instruction at `offset`.
///
/// Derived from the instruction's address rather than from a counter, so `emit` stays a pure
/// function of its arguments: two instructions in one block never share an address, so the ids
/// never collide and no state has to be threaded through the emitter.
fn fault_labels(offset: i64) -> (usize, usize) {
    let base = ((offset.max(0) as usize) / 4) * 2;
    (base, base + 1)
}

/// Guard the access whose host address is now in `rax`.
///
/// The access is allowed only if **every byte of it** lies inside `[mem_ptr, mem_ptr + mem_len)`.
/// Before this existed nothing checked at all: `mem_len` was written by the driver and read by
/// nobody, so a guest address outside the window read or wrote host memory. A benchmark never went
/// out of bounds, which is why it went unnoticed; an emulator hits it on the first unmapped page.
///
/// Emitted shape — check, then jump over a fault stub:
///
/// ```text
///   rcx = rax + (size-1)        ; last byte touched
///   cmp rcx, mem_ptr            ; jb  -> fault   (below the window)
///   cmp rcx, mem_ptr + mem_len  ; jae -> fault   (at or past the end)
///   jmp access
/// fault:
///   fault_addr = rax
///   pc = <this instruction>; return DataAbort
/// access:
/// ```
///
/// Register use is `rcx`, `rdx`, `rsi` — all caller-saved. Every call site loads the value operand
/// into `rsi` (or an xmm register) *after* the address is formed, so none of these three is live
/// across the guard. That invariant is why the guard can be inline instead of around a call, and it
/// is the thing to re-check if a future instruction is added that keeps a value in `rsi`.
fn mem_guard(c: &mut Code, size: i32, offset: i64) {
    let (fault, access) = fault_labels(offset);

    c.mov_reg(Reg::Rcx, Reg::Rax);
    if size > 1 {
        c.add_imm(Reg::Rcx, size - 1);
    }
    c.mov_load(Reg::Rdx, Reg::R15, state::MEM_PTR);
    c.cmp_reg(Reg::Rcx, Reg::Rdx);
    c.jcc(Cc::B, fault);
    c.mov_load(Reg::Rsi, Reg::R15, state::MEM_LEN);
    c.add_reg(Reg::Rdx, Reg::Rsi);
    c.cmp_reg(Reg::Rcx, Reg::Rdx);
    c.jcc(Cc::Ae, fault);
    c.jmp(access);
    c.label(fault);
    // Store the faulting address BEFORE the epilogue, which clobbers rax with the pc.
    c.mov_store(Reg::R15, state::FAULT_ADDR, Reg::Rax);
    epilogue(c, offset, Exit::DataAbort);
    c.label(access);
}

fn emit(c: &mut Code, instr: Instr, offset: i64) -> Result<(), Instr> {
    match instr {
        Instr::Nop => {}
        Instr::Mov {
            sf,
            op,
            rd,
            imm16,
            shift,
        } => {
            let v = u64::from(imm16);
            match op {
                MovOp::Zero => {
                    c.mov_imm64(Reg::Rax, v << shift);
                }
                MovOp::Not => {
                    c.mov_imm64(Reg::Rax, !(v << shift));
                    if !sf {
                        c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                        c.and_reg(Reg::Rax, Reg::Rcx);
                    }
                }
                MovOp::Keep => {
                    c.mov_load(Reg::Rax, Reg::R15, state::x(rd));
                    c.mov_imm64(Reg::Rcx, !(0xFFFFu64 << shift));
                    c.and_reg(Reg::Rax, Reg::Rcx);
                    c.mov_imm64(Reg::Rcx, v << shift);
                    c.or_reg(Reg::Rax, Reg::Rcx);
                }
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::LogicalImm {
            sf,
            op,
            rd,
            rn,
            n,
            immr,
            imms,
        } => {
            // The mask is known at compile time, so it goes in as an immediate.
            let imm = crate::interp::decode_bitmask(sf, n, immr, imms);
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_imm64(Reg::Rcx, imm);
            match op {
                LogicalOp::And | LogicalOp::Ands => c.and_reg(Reg::Rax, Reg::Rcx),
                LogicalOp::Orr => c.or_reg(Reg::Rax, Reg::Rcx),
                LogicalOp::Eor => c.xor_reg(Reg::Rax, Reg::Rcx),
            }
            if op == LogicalOp::Ands {
                materialise_flags(c, false);
            }
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::AddSubImm {
            sf,
            sub,
            set_flags,
            rd,
            rn,
            imm,
            lsl12,
        } => {
            let value = i32::try_from(imm << if lsl12 { 12 } else { 0 }).unwrap_or(i32::MAX);
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            if sub {
                c.mov_imm64(Reg::Rcx, u64::from(value as u32));
                c.sub_reg(Reg::Rax, Reg::Rcx);
            } else {
                c.add_imm(Reg::Rax, value);
            }
            if set_flags {
                materialise_flags(c, sub);
            }
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            // For ADD/SUB immediate, Rd == 31 addresses SP in the plain forms (`add sp, sp, #16`)
            // and is XZR in the flag-setting forms (`cmp` is SUBS with Rd == 31 and writes nothing).
            if !set_flags || rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::AddSubShifted {
            sf,
            sub,
            set_flags,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
            shift_reg(c, shift, amount);
            if sub {
                c.sub_reg(Reg::Rax, Reg::Rcx);
            } else {
                c.add_reg(Reg::Rax, Reg::Rcx);
            }
            if set_flags {
                materialise_flags(c, sub);
            }
            if !sf {
                c.mov_imm64(Reg::Rdx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rdx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::LogicalShifted {
            sf,
            op,
            invert,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
            shift_reg(c, shift, amount);
            if invert {
                c.not(Reg::Rcx);
            }
            match op {
                LogicalOp::And | LogicalOp::Ands => c.and_reg(Reg::Rax, Reg::Rcx),
                LogicalOp::Orr => c.or_reg(Reg::Rax, Reg::Rcx),
                LogicalOp::Eor => c.xor_reg(Reg::Rax, Reg::Rcx),
            }
            if op == LogicalOp::Ands {
                materialise_flags(c, false);
            }
            if !sf {
                c.mov_imm64(Reg::Rdx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rdx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::Bitfield {
            sf,
            signed,
            rd,
            rn,
            immr,
            imms,
        } => {
            // UBFM/SBFM: shift right by immr, then keep imms+1 bits (sign-extended for SBFM when
            // imms < width-1). The two cases (imms >= immr and the wrap case) are handled by
            // computing the same value the interpreter does, via shr then mask.
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            c.ror_imm(Reg::Rax, immr);
            let width: u32 = if sf { 64 } else { 32 };
            let size = u32::from(imms) + 1;
            let mask = if size >= 64 {
                u64::MAX
            } else {
                (1u64 << size) - 1
            };
            c.mov_imm64(Reg::Rcx, mask);
            c.and_reg(Reg::Rax, Reg::Rcx);
            if signed && size < width {
                // Sign-extend from bit size-1.
                c.mov_imm64(Reg::Rcx, 64 - size as u64);
                c.shift_cl(Reg::Rax, 4); // shl rax, cl
                c.shift_imm(Reg::Rax, 7, (64 - i64::from(size)) as u8); // sar rax, 64-size
            } else if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::MaddSub {
            sf,
            sub,
            rd,
            rn,
            rm,
            ra,
        } => {
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
            c.imul_reg(Reg::Rax, Reg::Rcx);
            if ra == 31 {
                if sub {
                    c.neg(Reg::Rax);
                }
            } else {
                c.mov_load(Reg::Rdx, Reg::R15, state::x(ra));
                if sub {
                    c.sub_reg(Reg::Rdx, Reg::Rax);
                    c.mov_reg(Reg::Rax, Reg::Rdx);
                } else {
                    c.add_reg(Reg::Rax, Reg::Rdx);
                }
            }
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::Csel {
            sf,
            rd,
            rn,
            rm,
            cond,
            op,
        } => {
            // result = cond ? rn : (rm with the op applied)
            //
            // The second arm goes in `rbx`, not `rcx`: materialising the condition needs four
            // scratch registers and clobbers rcx, which is exactly how the first version of this
            // emitted a select that returned the condition's own boolean.
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rbx, Reg::R15, state::x(rm));
            match op {
                1 => c.add_imm(Reg::Rbx, 1),
                2 => c.not(Reg::Rbx),
                3 => c.neg(Reg::Rbx),
                _ => {}
            }
            c.mov_reg(Reg::Rsi, Reg::Rax); // keep rn
            emit_cond_true(c, cond);
            c.test_reg(Reg::Rax, Reg::Rax);
            c.jcc(Cc::E, 4);
            c.jmp(5);
            c.label(4);
            c.mov_reg(Reg::Rsi, Reg::Rbx);
            c.label(5);
            c.mov_reg(Reg::Rax, Reg::Rsi);
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::ShiftReg {
            sf,
            kind,
            rd,
            rn,
            rm,
        } => {
            // x86 shifts take the count in cl and mask it to the operand width, which is exactly
            // A64's modulo-width rule — but only for 64-bit operands. A 32-bit guest shift needs
            // the count masked to 5 bits explicitly, because x86 would use 6.
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
            if !sf {
                c.mov_imm64(Reg::Rdx, 0x1F);
            } else {
                c.mov_imm64(Reg::Rdx, 0x3F);
            }
            c.and_reg(Reg::Rcx, Reg::Rdx);
            match kind {
                ShiftType::Lsl => c.shift_cl(Reg::Rax, 4),
                ShiftType::Lsr => c.shift_cl(Reg::Rax, 5),
                ShiftType::Asr => c.shift_cl(Reg::Rax, 7),
                ShiftType::Ror => c.shift_cl(Reg::Rax, 1),
            }
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::Div {
            sf,
            signed,
            rd,
            rn,
            rm,
        } => {
            // A64 division by zero yields zero instead of trapping, so the divisor is tested
            // before the divide and a zero result is synthesised. Doing it the other way round
            // would raise SIGFPE on the host and kill the emulator.
            let nonzero = 6;
            let done = 7;
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
            if !sf {
                c.mov_imm64(Reg::Rdx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rdx);
                c.and_reg(Reg::Rcx, Reg::Rdx);
            }
            c.test_reg(Reg::Rcx, Reg::Rcx);
            c.jcc(Cc::Ne, nonzero);
            c.mov_imm64(Reg::Rdx, 0);
            c.mov_reg(Reg::Rax, Reg::Rdx);
            c.jmp(done);
            c.label(nonzero);
            if signed {
                if !sf {
                    c.movsxd(Reg::Rax, Reg::Rax);
                    c.movsxd(Reg::Rcx, Reg::Rcx);
                }
                c.cqo();
                c.div(Reg::Rcx, true);
            } else {
                c.zero_rdx();
                c.div(Reg::Rcx, false);
            }
            c.label(done);
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::LoadStore {
            load,
            size,
            signed,
            rt,
            rn,
            mode,
        } => {
            // Address into rax: guest_base + offset.
            let (disp, writeback) = match mode {
                AddressMode::Offset { imm } | AddressMode::Unscaled { imm } => (imm, None),
                AddressMode::PreIndex { imm } => (imm, Some(Some(imm))),
                AddressMode::PostIndex { imm } => (0, Some(Some(imm))),
                AddressMode::RegisterOffset {
                    rm,
                    extend,
                    scaled_by_size,
                } => {
                    if extend != 1 && extend != 3 {
                        return Err(instr);
                    }
                    c.mov_load(Reg::Rcx, Reg::R15, state::x(rm));
                    if scaled_by_size {
                        c.shift_imm(
                            Reg::Rcx,
                            4,
                            size.trailing_zeros() as u8,
                        );
                    }
                    (0, Some(None))
                }
            };
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rdx, Reg::R15, state::MEM_PTR);
            c.add_reg(Reg::Rax, Reg::Rdx);
            if let AddressMode::RegisterOffset { .. } = mode {
                c.add_reg(Reg::Rax, Reg::Rcx);
            }
            if disp != 0 {
                c.add_imm(Reg::Rax, i32::try_from(disp).map_err(|_| instr)?);
            }
            mem_guard(c, size as i32, offset);
            if load {
                let tmp = Reg::Rsi;
                match (size, signed) {
                    (1, false) => c.movzx_u8_load(tmp, Reg::Rax, 0),
                    (1, true) => c.movsx_i8_load(tmp, Reg::Rax, 0),
                    (2, _) | (4, _) | (8, _) => c.mov_load(tmp, Reg::Rax, 0),
                    _ => return Err(instr),
                }
                c.mov_store(Reg::R15, state::x(rt), tmp);
            } else {
                c.mov_load(Reg::Rsi, Reg::R15, state::x(rt));
                c.mov_store(Reg::Rax, 0, Reg::Rsi);
            }
            // Pre/post-index write the base back; the register-offset form has no writeback.
            if let Some(Some(imm)) = writeback {
                c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
                c.add_imm(Reg::Rax, i32::try_from(imm).map_err(|_| instr)?);
                c.mov_store(Reg::R15, state::x(rn), Reg::Rax);
            }
        }
        Instr::B { offset: rel } => {
            epilogue(c, offset + rel, Exit::Jumped);
        }
        Instr::BCond { cond, offset: rel } => {
            let taken = 0;
            emit_cond_true(c, cond);
            c.test_reg(Reg::Rax, Reg::Rax);
            c.jcc(Cc::Ne, taken);
            // not taken: fall through with pc = next
            epilogue(c, offset + 4, Exit::Jumped);
            c.label(taken);
            epilogue(c, offset + rel, Exit::Jumped);
        }
        Instr::Cbz {
            sf,
            nonzero,
            rt,
            offset: rel,
        } => {
            let taken = 0;
            c.mov_load(Reg::Rax, Reg::R15, state::x(rt));
            if !sf {
                c.mov_imm64(Reg::Rcx, 0xFFFF_FFFF);
                c.and_reg(Reg::Rax, Reg::Rcx);
            }
            c.test_reg(Reg::Rax, Reg::Rax);
            c.jcc(if nonzero { Cc::Ne } else { Cc::E }, taken);
            epilogue(c, offset + 4, Exit::Jumped);
            c.label(taken);
            epilogue(c, offset + rel, Exit::Jumped);
        }
        Instr::Tbz {
            nonzero,
            rt,
            bit,
            offset: rel,
        } => {
            // Test one bit of the register: shift it down to bit 0, mask, then branch.
            let taken = 8;
            let done = 9;
            c.mov_load(Reg::Rax, Reg::R15, state::x(rt));
            if bit != 0 {
                c.shift_imm(Reg::Rax, 5, bit);
            }
            c.mov_imm64(Reg::Rcx, 1);
            c.and_reg(Reg::Rax, Reg::Rcx);
            c.test_reg(Reg::Rax, Reg::Rax);
            c.jcc(if nonzero { Cc::Ne } else { Cc::E }, taken);
            epilogue(c, offset + 4, Exit::Jumped);
            c.label(taken);
            epilogue(c, offset + rel, Exit::Jumped);
            c.label(done);
        }
        Instr::Adr { rd, offset: rel } => {
            c.mov_imm64(Reg::Rax, (offset + rel) as u64);
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::Adrp { rd, page_offset } => {
            // The immediate is a page number relative to the instruction's own page.
            let base = (offset as u64) & !0xFFFu64;
            c.mov_imm64(Reg::Rax, base.wrapping_add_signed(page_offset));
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::Svc { .. } => {
            epilogue(c, offset, Exit::SupervisorCall);
        }
        Instr::Ret { rn } => {
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_store(Reg::R15, state::PC, Reg::Rax);
            epilogue_tail(c, Exit::Jumped);
        }
        Instr::Br { rn } | Instr::Blr { rn } => {
            if matches!(instr, Instr::Blr { .. }) {
                // The return address is the next instruction, architecturally.
                c.mov_imm64(Reg::Rax, (offset + 4) as u64);
                c.mov_store(Reg::R15, state::x(30), Reg::Rax);
            }
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_store(Reg::R15, state::PC, Reg::Rax);
            epilogue_tail(c, Exit::Jumped);
        }
        Instr::Bl { offset: rel } => {
            c.mov_imm64(Reg::Rax, (offset + 4) as u64);
            c.mov_store(Reg::R15, state::x(30), Reg::Rax);
            epilogue(c, offset + rel, Exit::Jumped);
        }
        Instr::LoadStorePair {
            load,
            sf,
            rt,
            rt2,
            rn,
            mode,
        } => {
            let size: u8 = if sf { 8 } else { 4 };
            let (disp, writeback) = match mode {
                AddressMode::Offset { imm } => (imm, None),
                AddressMode::PreIndex { imm } => (imm, Some(imm)),
                AddressMode::PostIndex { imm } => (0, Some(imm)),
                _ => return Err(instr),
            };
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rdx, Reg::R15, state::MEM_PTR);
            c.add_reg(Reg::Rax, Reg::Rdx);
            if disp != 0 {
                c.add_imm(Reg::Rax, i32::try_from(disp).map_err(|_| instr)?);
            }
            mem_guard(c, size as i32 * 2, offset);
            let second = i32::from(size);
            if load {
                c.mov_load(Reg::Rsi, Reg::Rax, 0);
                if rt != 31 {
                    c.mov_store(Reg::R15, state::x(rt), Reg::Rsi);
                }
                c.mov_load(Reg::Rsi, Reg::Rax, second);
                if rt2 != 31 {
                    c.mov_store(Reg::R15, state::x(rt2), Reg::Rsi);
                }
            } else {
                c.mov_load(Reg::Rsi, Reg::R15, state::x(rt));
                c.mov_store(Reg::Rax, 0, Reg::Rsi);
                c.mov_load(Reg::Rsi, Reg::R15, state::x(rt2));
                c.mov_store(Reg::Rax, second, Reg::Rsi);
            }
            if let Some(imm) = writeback {
                c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
                c.add_imm(Reg::Rax, i32::try_from(imm).map_err(|_| instr)?);
                c.mov_store(Reg::R15, state::x(rn), Reg::Rax);
            }
        }
        // Scalar double-precision. Single-precision forms are refused rather than computed at the
        // wrong width.
        Instr::FpArith { dp: true, op, rd, rn, rm } => {
            c.movsd_load(0, Reg::R15, state::fp(rn));
            c.movsd_load(1, Reg::R15, state::fp(rm));
            let opcode = match op {
                FpOp::Add => 0x58,
                FpOp::Sub => 0x5C,
                FpOp::Mul => 0x59,
                FpOp::Div => 0x5E,
                FpOp::Neg => return Err(instr),
            };
            c.sd_arith(opcode, 0, 1);
            c.movsd_store(Reg::R15, state::fp(rd), 0);
        }
        Instr::FpFused {
            dp: true,
            op,
            rd,
            rn,
            rm,
            ra,
        } => {
            // acc +/- (a * b), then negate the result for the N forms.
            c.movsd_load(0, Reg::R15, state::fp(rn));
            c.movsd_load(1, Reg::R15, state::fp(rm));
            c.sd_arith(0x59, 0, 1); // mulsd
            c.movsd_load(1, Reg::R15, state::fp(ra));
            c.sd_arith(if op & 1 != 0 { 0x5C } else { 0x58 }, 1, 0); // acc +/- product
            c.movsd_store(Reg::R15, state::fp(rd), 1);
            if op & 2 != 0 {
                // Negate: xor the sign bit. Load, flip, store.
                c.movq_from_xmm(Reg::Rax, 1);
                c.mov_imm64(Reg::Rcx, 1u64 << 63);
                c.xor_reg(Reg::Rax, Reg::Rcx);
                c.movq_to_xmm(1, Reg::Rax);
                c.movsd_store(Reg::R15, state::fp(rd), 1);
            }
        }
        Instr::FpMov { dp: true, rd, rn } => {
            c.movsd_load(0, Reg::R15, state::fp(rn));
            c.movsd_store(Reg::R15, state::fp(rd), 0);
        }
        Instr::FpMovImm { dp, rd, imm8 } => {
            let bits = crate::interp::vfp_expand_imm(imm8, dp);
            c.mov_imm64(Reg::Rax, bits);
            c.movq_to_xmm(0, Reg::Rax);
            c.movsd_store(Reg::R15, state::fp(rd), 0);
        }
        Instr::FpToGen { dp: true, rd, rn } => {
            c.movsd_load(0, Reg::R15, state::fp(rn));
            c.movq_from_xmm(Reg::Rax, 0);
            if rd != 31 {
                c.mov_store(Reg::R15, state::x(rd), Reg::Rax);
            }
        }
        Instr::GenToFp { dp: true, rd, rn } => {
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.movq_to_xmm(0, Reg::Rax);
            c.movsd_store(Reg::R15, state::fp(rd), 0);
        }
        Instr::FpLoadStore {
            load,
            size,
            rt,
            rn,
            mode,
        } => {
            if size != 8 {
                return Err(instr);
            }
            let (disp, writeback) = match mode {
                AddressMode::Offset { imm } | AddressMode::Unscaled { imm } => (imm, None),
                AddressMode::PreIndex { imm } => (imm, Some(imm)),
                AddressMode::PostIndex { imm } => (0, Some(imm)),
                _ => return Err(instr),
            };
            c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
            c.mov_load(Reg::Rdx, Reg::R15, state::MEM_PTR);
            c.add_reg(Reg::Rax, Reg::Rdx);
            if disp != 0 {
                c.add_imm(Reg::Rax, i32::try_from(disp).map_err(|_| instr)?);
            }
            mem_guard(c, 8, offset);
            if load {
                c.movsd_load(0, Reg::Rax, 0);
                c.movsd_store(Reg::R15, state::fp(rt), 0);
            } else {
                c.movsd_load(0, Reg::R15, state::fp(rt));
                c.movsd_store(Reg::Rax, 0, 0);
            }
            if let Some(imm) = writeback {
                c.mov_load(Reg::Rax, Reg::R15, state::x(rn));
                c.add_imm(Reg::Rax, i32::try_from(imm).map_err(|_| instr)?);
                c.mov_store(Reg::R15, state::x(rn), Reg::Rax);
            }
        }
        // Everything else is the driver's problem, not a silent no-op.
        _ => return Err(instr),
    }
    Ok(())
}

/// `shl`/`shr`/`sar`/`ror` of `rcx` by an immediate.
fn shift_reg(c: &mut Code, kind: ShiftType, amount: u8) {
    match kind {
        ShiftType::Lsl => c.shift_imm(Reg::Rcx, 4, amount),
        ShiftType::Lsr => c.shift_imm(Reg::Rcx, 5, amount),
        ShiftType::Asr => c.shift_imm(Reg::Rcx, 7, amount),
        ShiftType::Ror => c.ror_imm(Reg::Rcx, amount),
    }
}

/// Turn the current x86 flags into the packed A64 flags byte.
///
/// N, Z, S and O map directly; C is the same bit as x86's CF because A64's subtraction carry is
/// not-borrow, which is what x86's `CF` already means after `SUB`/`CMP`.
fn materialise_flags(c: &mut Code, subtraction: bool) {
    // Snapshot ALL FOUR flags first. `shl`/`or` and friends set flags themselves, so combining as
    // we go reads the wrong ones: the first version of this function shifted N into place and then
    // asked `setcc(Ae)` for C, which answered about the shift, not the subtraction. It also walked
    // over `rax`, which held the result.
    c.setcc(Cc::S, Reg::R8);
    c.setcc(Cc::E, Reg::R9);
    // x86 CF is *carry* after ADD and *borrow* after SUB; A64's C is carry in both cases. So the
    // carry bit is `CF` for an add and `not CF` for a subtract — a distinction that is invisible
    // until a `CS`/`CC`/`HI`/`LS` branch is taken on the wrong side.
    c.setcc(if subtraction { Cc::Ae } else { Cc::B }, Reg::R10);
    c.setcc(Cc::O, Reg::R11);

    // Now combine, in registers the flags no longer matter for.
    c.movzx_u8(Reg::Rcx, Reg::R8); // N
    c.shift_imm(Reg::Rcx, 4, 3);
    c.movzx_u8(Reg::Rdx, Reg::R9); // Z
    c.shift_imm(Reg::Rdx, 4, 2);
    c.or_reg(Reg::Rcx, Reg::Rdx);
    c.movzx_u8(Reg::Rdx, Reg::R10); // C
    c.shift_imm(Reg::Rdx, 4, 1);
    c.or_reg(Reg::Rcx, Reg::Rdx);
    c.movzx_u8(Reg::Rdx, Reg::R11); // V
    c.or_reg(Reg::Rcx, Reg::Rdx);
    c.mov_store(Reg::R15, state::FLAGS, Reg::Rcx);
}




/// Exit without touching `pc` — for paths that have already stored it.
fn epilogue_tail(c: &mut Code, exit: Exit) {
    c.mov_imm64(Reg::Rax, exit as u64);
    c.pop(Reg::R15);
    c.pop(Reg::Rbx);
    c.ret();
}

/// Emit the standard exit: write `pc`, put the exit code in `rax`, restore the callee-saved
/// registers and return.
///
/// Restoring `r15` is not optional. It holds the state pointer for the body of the function, but it
/// is **callee-saved** in the System V ABI: the Rust caller may be using it for its own locals, and
/// returning with it clobbered corrupted the caller's stack frame — which surfaced as glibc's
/// `free(): invalid size` several calls later, nowhere near the generated code.
fn epilogue(c: &mut Code, pc: i64, exit: Exit) {
    c.mov_imm64(Reg::Rax, pc as u64);
    c.mov_store(Reg::R15, state::PC, Reg::Rax);
    c.mov_imm64(Reg::Rax, exit as u64);
    c.pop(Reg::R15);
    c.pop(Reg::Rbx);
    c.ret();
}

/// Emit code that leaves 1 in `rax` when `cond` holds, 0 otherwise, reading the packed flags byte
/// from memory.
///
/// Generated code never trusts the live x86 flags for a guest branch: the flag-packing sequence
/// clobbers them, and x86's carry means different things after an add and after a subtract. The
/// packed byte is the single source of truth, and every consumer reads it.
fn emit_cond_true(c: &mut Code, cond: Cond) {
    c.movzx_u8_load(Reg::Rax, Reg::R15, state::FLAGS);
    // rcx = N, rdx = Z, r8 = C, r9 = V
    c.mov_reg(Reg::Rcx, Reg::Rax);
    c.shift_imm(Reg::Rcx, 5, 3);
    c.mov_reg(Reg::Rdx, Reg::Rax);
    c.shift_imm(Reg::Rdx, 5, 2);
    c.mov_reg(Reg::R8, Reg::Rax);
    c.shift_imm(Reg::R8, 5, 1);
    c.mov_reg(Reg::R9, Reg::Rax);
    c.mov_imm64(Reg::R10, 1);
    c.and_reg(Reg::R9, Reg::R10);
    c.and_reg(Reg::R8, Reg::R10);
    c.and_reg(Reg::Rdx, Reg::R10);
    c.and_reg(Reg::Rcx, Reg::R10);

    match cond {
        Cond::Eq => c.mov_reg(Reg::Rax, Reg::Rdx),
        Cond::Ne => {
            c.mov_reg(Reg::Rax, Reg::Rdx);
            c.xor_reg(Reg::Rax, Reg::R10);
        }
        Cond::Cs => c.mov_reg(Reg::Rax, Reg::R8),
        Cond::Cc => {
            c.mov_reg(Reg::Rax, Reg::R8);
            c.xor_reg(Reg::Rax, Reg::R10);
        }
        Cond::Mi => c.mov_reg(Reg::Rax, Reg::Rcx),
        Cond::Pl => {
            c.mov_reg(Reg::Rax, Reg::Rcx);
            c.xor_reg(Reg::Rax, Reg::R10);
        }
        Cond::Vs => c.mov_reg(Reg::Rax, Reg::R9),
        Cond::Vc => {
            c.mov_reg(Reg::Rax, Reg::R9);
            c.xor_reg(Reg::Rax, Reg::R10);
        }
        Cond::Hi => {
            // C && !Z
            c.mov_reg(Reg::Rax, Reg::Rdx);
            c.xor_reg(Reg::Rax, Reg::R10);
            c.and_reg(Reg::Rax, Reg::R8);
        }
        Cond::Ls => {
            // !C || Z
            c.mov_reg(Reg::Rax, Reg::R8);
            c.xor_reg(Reg::Rax, Reg::R10);
            c.or_reg(Reg::Rax, Reg::Rdx);
        }
        Cond::Ge => {
            // N == V
            c.mov_reg(Reg::Rax, Reg::Rcx);
            c.xor_reg(Reg::Rax, Reg::R9);
            c.xor_reg(Reg::Rax, Reg::R10);
        }
        Cond::Lt => {
            // N != V
            c.mov_reg(Reg::Rax, Reg::Rcx);
            c.xor_reg(Reg::Rax, Reg::R9);
        }
        Cond::Gt => {
            // !Z && (N == V)
            c.mov_reg(Reg::R11, Reg::Rcx);
            c.xor_reg(Reg::R11, Reg::R9);
            c.xor_reg(Reg::R11, Reg::R10); // N==V
            c.mov_reg(Reg::Rax, Reg::Rdx);
            c.xor_reg(Reg::Rax, Reg::R10); // !Z
            c.and_reg(Reg::Rax, Reg::R11);
        }
        Cond::Le => {
            // Z || (N != V)
            c.mov_reg(Reg::R11, Reg::Rcx);
            c.xor_reg(Reg::R11, Reg::R9); // N!=V
            c.mov_reg(Reg::Rax, Reg::Rdx);
            c.or_reg(Reg::Rax, Reg::R11);
        }
        Cond::Al | Cond::Nv => c.mov_imm64(Reg::Rax, 1),
    }
}

/// An executable mapping holding one compiled block.
///
/// `mmap` with PROT_READ|PROT_WRITE|PROT_EXEC, because a W^X split would need a second mapping and
/// a copy; this project is not defending against itself, and the alternative is a debugger-hostile
/// arrangement that buys nothing here.
pub struct Exec {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the pointer is an anonymous mapping owned exclusively by this value; it is never aliased
// mutably elsewhere, and the generated code only touches the state struct it is handed.
unsafe impl Send for Exec {}

impl Exec {
    /// Copy `code` into a fresh executable mapping.
    pub fn new(code: &[u8]) -> Self {
        let page = 4096usize;
        let len = code.len().div_ceil(page) * page;
        assert!(len > 0, "cannot map an empty block");
        // SAFETY: anonymous private mapping of `len` bytes; checked against MAP_FAILED below.
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
        // SAFETY: the mapping is `len` bytes, writable, and `code` is shorter than or equal to it.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
        }
        Self {
            ptr: ptr.cast::<u8>(),
            len,
        }
    }

    /// The generated function.
    pub fn entry(&self) -> extern "C" fn(*mut JitState) -> u64 {
        // SAFETY: `new` was given exactly one compiled block for this signature.
        unsafe { std::mem::transmute::<*mut u8, extern "C" fn(*mut JitState) -> u64>(self.ptr) }
    }

    /// First bytes of the mapping, for a disassembly check.
    pub fn as_slice(&self, len: usize) -> &[u8] {
        // SAFETY: the mapping is at least `len` bytes only when the caller passes a length no
        // larger than the code that was copied in; `new` cannot know it, so clamp to the page size.
        unsafe { std::slice::from_raw_parts(self.ptr, len.min(self.len)) }
    }
}

impl Drop for Exec {
    fn drop(&mut self) {
        // SAFETY: the pointer and length are exactly those returned by mmap in `new`, unmapped once.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

/// A guest-address to compiled-code map.
///
/// Without this, every block entry pays a translation plus an `mmap`, which makes a JIT roughly a
/// hundred times slower than the interpreter — measured, not guessed. The cache is the difference
/// between the two designs, and it is keyed on the guest address the block starts at.
pub struct BlockCache {
    /// Start address -> (compiled code, guest address one past the block's last instruction).
    ///
    /// The end is stored because invalidation has to drop a block that *overlaps* a range, not only
    /// one that starts inside it: a branch discovered into the middle of a long block makes that
    /// block wrong, and its start is outside the range being invalidated.
    blocks: std::collections::HashMap<u64, (Exec, u64)>,
}

impl BlockCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self {
            blocks: std::collections::HashMap::new(),
        }
    }

    /// How many blocks are compiled and resident.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// True when nothing has been compiled.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Compile `block` (which must start at `start`) if it is not already cached, and return its
    /// entry point.
    pub fn get_or_compile(&mut self, start: u64, block: &crate::block::Block) -> Result<extern "C" fn(*mut JitState) -> u64, Instr> {
        if let Some((exec, _end)) = self.blocks.get(&start) {
            return Ok(exec.entry());
        }
        let code = compile(block)?;
        let exec = Exec::new(&code);
        let entry = exec.entry();
        self.blocks.insert(start, (exec, block.end as u64));
        Ok(entry)
    }
}

impl BlockCache {
    /// Drop every compiled block that OVERLAPS `[addr, addr + size)`; returns how many went.
    ///
    /// Overlap, not containment: a block that starts before the range and runs into it is stale the
    /// moment the range changes, and a start-only test would leave it resident and wrong. The
    /// returned count is there so a caller can tell "nothing matched" from "the range was wrong" —
    /// a silent zero is exactly how a stale block survives an invalidation and then executes old
    /// code at a new address.
    pub fn invalidate_range(&mut self, addr: u64, size: u64) -> usize {
        let end = addr.saturating_add(size);
        let before = self.blocks.len();
        self.blocks.retain(|&start, (_exec, guest_end)| {
            let overlaps = start < end && *guest_end > addr;
            !overlaps
        });
        before - self.blocks.len()
    }
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A JIT that fetches blocks from guest memory **by address**.
///
/// `run_cached` takes the whole program as a `&[u32]`: right for a benchmark, useless to an
/// emulator, whose instructions live behind a page table and whose blocks are only discovered as
/// execution reaches them. Here words arrive through `read_word` (which answers `None` for an
/// address it cannot read) and blocks are cached by guest address, which is also what makes
/// [`BlockCache::invalidate_range`] meaningful.
pub struct AddressJit<F>
where
    F: FnMut(u64) -> Option<u32>,
{
    read_word: F,
    cache: BlockCache,
    state: JitState,
    /// Bound on one block's decode. Real code terminates blocks in a handful of instructions; this
    /// exists so a page of zeros or garbage cannot spin the scanner forever.
    max_instructions: usize,
}

impl<F> AddressJit<F>
where
    F: FnMut(u64) -> Option<u32>,
{
    /// Build a JIT over `read_word`, starting from `state`.
    pub fn new(read_word: F, state: JitState) -> Self {
        Self {
            read_word,
            cache: BlockCache::new(),
            state,
            max_instructions: 1024,
        }
    }

    /// The guest state. Registers are read and written here, not through accessors.
    pub fn state(&self) -> &JitState {
        &self.state
    }

    /// Mutable guest state, so a caller can load registers and `pc` before a run.
    pub fn state_mut(&mut self) -> &mut JitState {
        &mut self.state
    }

    /// How many blocks are compiled and resident.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Drop cached translations overlapping `[addr, addr + size)`. Returns the count dropped.
    pub fn invalidate(&mut self, addr: u64, size: u64) -> usize {
        self.cache.invalidate_range(addr, size)
    }

    /// Execute exactly ONE guest instruction, translated fresh and NOT cached.
    ///
    /// Block-granular stepping is not stepping: a debugger that retires a whole basic block per
    /// "step" has already moved past the instruction the user was looking at. The translation is
    /// deliberately uncached — a one-instruction block at the same address would collide in the
    /// cache with the real block that belongs there.
    pub fn step(&mut self) -> Result<u64, String> {
        let pc = self.state.pc;
        if !pc.is_multiple_of(4) {
            return Err(format!("misaligned pc {pc:#x}"));
        }
        let block = {
            let reader = &mut self.read_word;
            crate::block::scan_at(pc as i64, |a: i64| reader(a as u64), 1)
        };
        let Some(block) = block else {
            return Err(format!("nothing mapped at {pc:#x}"));
        };
        if !block.is_compilable() {
            return Err(format!("untranslatable word at {pc:#x}"));
        }
        let code = compile(&block).map_err(|i| format!("unsupported instruction: {i:?}"))?;
        let exec = Exec::new(&code);
        let exit = (exec.entry())(&mut self.state);
        self.state.last_exit = exit;
        Ok(1)
    }

    /// Run until a supervisor call, a data abort, an unsupported instruction, or `max_blocks`.
    ///
    /// The exit reason is left in `state.last_exit`; `state.pc` is where execution stopped, which on
    /// a data abort is the faulting instruction itself, so the caller can map the page and re-run.
    pub fn run(&mut self, max_blocks: u64) -> Result<u64, String> {
        let mut executed = 0u64;
        while executed < max_blocks {
            let pc = self.state.pc;
            if !pc.is_multiple_of(4) {
                return Err(format!("misaligned pc {pc:#x}"));
            }
            let block = {
                let reader = &mut self.read_word;
                crate::block::scan_at(pc as i64, |a: i64| reader(a as u64), self.max_instructions)
            };
            let Some(block) = block else {
                return Err(format!("nothing mapped at {pc:#x}"));
            };
            if !block.is_compilable() {
                return Err(format!("untranslatable block at {pc:#x}"));
            }
            let entry = self
                .cache
                .get_or_compile(pc, &block)
                .map_err(|i| format!("unsupported instruction: {i:?}"))?;
            let exit = entry(&mut self.state);
            executed += 1;
            // Written on EVERY exit, not only on the two interesting ones. Setting it only for
            // SVC and DataAbort left a normal block exit reporting 0 forever, which reads as
            // "fell off the end" and is indistinguishable from an uninitialised field.
            self.state.last_exit = exit;
            if exit == Exit::SupervisorCall as u64 || exit == Exit::DataAbort as u64 {
                return Ok(executed);
            }
            if exit == Exit::Unsupported as u64 {
                return Err("unsupported instruction".into());
            }
            if exit != Exit::Next as u64 && exit != Exit::Jumped as u64 {
                return Err(format!("bad exit code {exit}"));
            }
        }
        Ok(executed)
    }
}

/// Compile and run blocks from `state.pc` until a supervisor call, an unsupported instruction, or
/// `max_blocks` blocks have retired.
///
/// Returns the number of blocks executed. Blocks are compiled fresh each time they are entered:
/// this is a driver for testing and for measuring the translation rate, not a cache. A real
/// runtime keeps a map from guest address to compiled block, which is the next piece of work.
pub fn run(
    words: &[u32],
    memory: &mut [u8],
    state: &mut JitState,
    max_blocks: u64,
) -> Result<u64, String> {
    let mut cache = BlockCache::new();
    run_cached(words, memory, state, max_blocks, &mut cache)
}

/// The same driver, reusing a caller-owned cache across calls.
pub fn run_cached(
    words: &[u32],
    memory: &mut [u8],
    state: &mut JitState,
    max_blocks: u64,
    cache: &mut BlockCache,
) -> Result<u64, String> {
    state.mem_ptr = memory.as_mut_ptr() as u64;
    state.mem_len = memory.len() as u64;

    // Decode the whole region ONCE. The first version of this driver called `block::scan` inside the
    // loop, which re-decoded everything from the current pc to the end on every block entry — the
    // single largest cost in the benchmark, and entirely avoidable.
    let program = crate::block::scan(words);

    let mut executed = 0u64;
    while executed < max_blocks {
        let pc = state.pc as usize;
        if pc >= words.len() * 4 {
            return Ok(executed);
        }
        executed += 1;
        let block = program
            .block_at(pc as i64)
            .ok_or_else(|| format!("no block at {pc:#x}"))?;

        let f = cache
            .get_or_compile(pc as u64, block)
            .map_err(|i| format!("unsupported: {i:?}"))?;
        // SAFETY: `f` was compiled with this signature and `state` outlives the call.
        let exit = f(state);
        match exit {
            x if x == Exit::Next as u64 || x == Exit::Jumped as u64 => {}
            x if x == Exit::SupervisorCall as u64 => {
                state.last_exit = x;
                return Ok(executed);
            }
            x if x == Exit::DataAbort as u64 => {
                state.last_exit = x;
                return Ok(executed);
            }
            x if x == Exit::Unsupported as u64 => return Err("unsupported instruction".into()),
            other => return Err(format!("bad exit code {other}")),
        }
        // No "did the pc move?" check: a self-branch (`cbnz x1, #-8` targeting its own block
        // start) legitimately returns to the same address. The block budget is the bound.
    }
    Ok(executed)
}
