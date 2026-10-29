//! A reference interpreter for the decoded subset.
//!
//! This is not the fast path and is not meant to be. It exists so that the recompiler has an
//! oracle: when the JIT and the interpreter disagree on the same block, one of them is wrong, and
//! the interpreter is simple enough to be checked by eye.
//!
//! It is also the thing that makes differential testing against `qemu-aarch64` possible: the same
//! machine code, run here and there, must reach the same architectural state.
//!
//! # Scope and honesty about it
//!
//! Implemented: the instructions [`crate::a64::decode`] understands, minus the floating-point set,
//! which is decoded but not executed (it returns [`Trap::Unsupported`] rather than a wrong answer).
//! Memory is a flat byte array with a single base; there is no MMU, no cache and no alignment trap.
//!
//! Flags are computed the way A64 defines them, including the subtle ones: `C` on subtraction is
//! the *not-borrow* flag, and `V` is signed overflow, which is not the same thing as the sign of
//! the result. Those two are where hand-written interpreters usually drift.

use crate::a64::{self, AddressMode, Cond, Instr, LogicalOp, MovOp, ShiftType};

/// What went wrong while executing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    /// A decoded instruction this interpreter does not implement.
    Unsupported,
    /// An instruction word the decoder could not identify.
    Undecoded { pc: u64, word: u32 },
    /// A memory access outside the flat memory model.
    BadAddress { address: u64, size: u8 },
    /// An `SVC`. The caller decides what a supervisor call means.
    SupervisorCall { imm: u16 },
    /// The program counter left the mapped region.
    PcOutOfRange { pc: u64 },
    /// An instruction budget was exhausted; the program is probably looping forever.
    StepLimit { steps: u64 },
}

/// Architectural state.
#[derive(Debug, Clone)]
pub struct Machine {
    /// `X0..X30`. Reads of W registers use the low 32 bits and zero the rest.
    pub x: [u64; 31],
    /// Stack pointer. Kept apart from the general registers, as A64 does.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// Condition flags.
    pub n: bool,
    /// Zero flag.
    pub z: bool,
    /// Carry / not-borrow.
    pub c: bool,
    /// Signed overflow.
    pub v: bool,
    /// Flat memory. Address `base + i` maps to `memory[i]`.
    pub memory: Vec<u8>,
    /// Address that `memory[0]` corresponds to.
    pub base: u64,
}

impl Machine {
    /// A machine with `bytes` of zeroed memory mapped at `base`.
    pub fn new(base: u64, bytes: usize) -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: base,
            n: false,
            z: false,
            c: false,
            v: false,
            memory: vec![0; bytes],
            base,
        }
    }

    /// Load instruction words at `address`, returning the address after them.
    pub fn load_code(&mut self, address: u64, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            let off = (address + (i as u64) * 4 - self.base) as usize;
            self.memory[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    fn index(&self, address: u64, size: u8) -> Result<usize, Trap> {
        let rel = address.checked_sub(self.base).ok_or(Trap::BadAddress {
            address,
            size,
        })?;
        let end = rel + u64::from(size);
        if end > self.memory.len() as u64 {
            return Err(Trap::BadAddress { address, size });
        }
        Ok(rel as usize)
    }

    /// Read `size` bytes little-endian from guest memory.
    pub fn read(&self, address: u64, size: u8) -> Result<u64, Trap> {
        let i = self.index(address, size)?;
        let mut buf = [0u8; 8];
        buf[..size as usize].copy_from_slice(&self.memory[i..i + size as usize]);
        Ok(u64::from_le_bytes(buf))
    }

    /// Write the low `size` bytes of `value` to guest memory.
    pub fn write(&mut self, address: u64, size: u8, value: u64) -> Result<(), Trap> {
        let i = self.index(address, size)?;
        self.memory[i..i + size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
        Ok(())
    }

    /// Read a general register. **Register 31 reads as zero here** — that is XZR, and it is the
    /// right default: only `ADD`/`SUB` immediate and a load/store base treat 31 as SP, and those
    /// go through [`Machine::get_sp`].
    fn get(&self, r: u8) -> u64 {
        if r == 31 { 0 } else { self.x[r as usize] }
    }

    /// Write a general register; a write to 31 is discarded (XZR).
    fn set(&mut self, r: u8, value: u64) {
        if r != 31 {
            self.x[r as usize] = value;
        }
    }

    /// Read a register where 31 means SP.
    fn get_sp(&self, r: u8) -> u64 {
        if r == 31 { self.sp } else { self.x[r as usize] }
    }

    /// Write a register where 31 means SP.
    fn set_sp(&mut self, r: u8, value: u64) {
        if r == 31 {
            self.sp = value;
        } else {
            self.x[r as usize] = value;
        }
    }

    /// Read a register at a given operand width, honouring W-register zero-extension.
    fn get_sized(&self, r: u8, sf: bool) -> u64 {
        let v = self.get(r);
        if sf { v } else { v & 0xFFFF_FFFF }
    }

    /// Write a register at a given operand width, zero-extending a W result.
    fn set_sized(&mut self, r: u8, sf: bool, value: u64) {
        let v = if sf { value } else { value & 0xFFFF_FFFF };
        self.set(r, v);
    }

    fn cond_holds(&self, cond: Cond) -> bool {
        match cond {
            Cond::Eq => self.z,
            Cond::Ne => !self.z,
            Cond::Cs => self.c,
            Cond::Cc => !self.c,
            Cond::Mi => self.n,
            Cond::Pl => !self.n,
            Cond::Vs => self.v,
            Cond::Vc => !self.v,
            Cond::Hi => self.c && !self.z,
            Cond::Ls => !self.c || self.z,
            Cond::Ge => self.n == self.v,
            Cond::Lt => self.n != self.v,
            Cond::Gt => !self.z && self.n == self.v,
            Cond::Le => self.z || self.n != self.v,
            Cond::Al | Cond::Nv => true,
        }
    }

    fn shift(value: u64, kind: ShiftType, amount: u8, sf: bool) -> u64 {
        let width = if sf { 64 } else { 32 };
        let amount = u32::from(amount) % width;
        let v = if sf { value } else { value & 0xFFFF_FFFF };
        let out = match kind {
            ShiftType::Lsl => v.wrapping_shl(amount),
            ShiftType::Lsr => v.wrapping_shr(amount),
            ShiftType::Asr => {
                if sf {
                    ((v as i64) >> amount) as u64
                } else {
                    (((v as u32 as i32) >> amount) as u32) as u64
                }
            }
            ShiftType::Ror => {
                if amount == 0 {
                    v
                } else if sf {
                    v.rotate_right(amount)
                } else {
                    (v as u32).rotate_right(amount) as u64
                }
            }
        };
        if sf { out } else { out & 0xFFFF_FFFF }
    }

    /// Execute one instruction at `pc`, advancing `pc`. Returns a trap for anything unimplemented.
    pub fn step(&mut self) -> Result<(), Trap> {
        let word = self.read(self.pc, 4)? as u32;
        let instr = a64::decode(word);
        let pc = self.pc;
        let next = pc.wrapping_add(4);

        match instr {
            Instr::Unknown { word } => return Err(Trap::Undecoded { pc, word }),
            Instr::Nop => self.pc = next,
            Instr::Adr { rd, offset } => {
                self.set(rd, pc.wrapping_add_signed(offset));
                self.pc = next;
            }
            Instr::Adrp { rd, page_offset } => {
                self.set(rd, (pc & !0xFFF).wrapping_add_signed(page_offset));
                self.pc = next;
            }
            Instr::B { offset } => self.pc = pc.wrapping_add_signed(offset),
            Instr::Bl { offset } => {
                self.set(30, next);
                self.pc = pc.wrapping_add_signed(offset);
            }
            Instr::BCond { cond, offset } => {
                self.pc = if self.cond_holds(cond) {
                    pc.wrapping_add_signed(offset)
                } else {
                    next
                };
            }
            Instr::Cbz {
                sf,
                nonzero,
                rt,
                offset,
            } => {
                let zero = self.get_sized(rt, sf) == 0;
                let take = zero != nonzero;
                self.pc = if take { pc.wrapping_add_signed(offset) } else { next };
            }
            Instr::Tbz {
                nonzero,
                rt,
                bit,
                offset,
            } => {
                let set = (self.get(rt) >> bit) & 1 != 0;
                let take = set == nonzero;
                self.pc = if take { pc.wrapping_add_signed(offset) } else { next };
            }
            Instr::Br { rn } => self.pc = self.get(rn),
            Instr::Blr { rn } => {
                let target = self.get(rn);
                self.set(30, next);
                self.pc = target;
            }
            Instr::Ret { rn } => self.pc = self.get(rn),
            Instr::Mov {
                sf,
                op,
                rd,
                imm16,
                shift,
            } => {
                let v = u64::from(imm16) << shift;
                let v = match op {
                    MovOp::Zero => v,
                    MovOp::Keep => {
                        let mask = !(0xFFFFu64 << shift);
                        (self.get_sized(rd, sf) & mask) | v
                    }
                    MovOp::Not => !v,
                };
                self.set_sized(rd, sf, v);
                self.pc = next;
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
                let imm = u64::from(imm) << if lsl12 { 12 } else { 0 };
                // ADD/SUB immediate is one of the few encodings where 31 means SP.
                let a = if sf { self.get_sp(rn) } else { self.get_sp(rn) & 0xFFFF_FFFF };
                let (result, carry, overflow) = if sub {
                    add_with_carry(a, !imm, 1, sf)
                } else {
                    add_with_carry(a, imm, 0, sf)
                };
                if !set_flags || rd != 31 {
                    let v = if sf { result } else { result & 0xFFFF_FFFF };
                    self.set_sp(rd, v);
                }
                if set_flags {
                    self.set_flags(result, carry, overflow, sf);
                }
                self.pc = next;
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
                let b = Self::shift(self.get_sized(rm, sf), shift, amount, sf);
                let a = self.get_sized(rn, sf);
                let (result, carry, overflow) = if sub {
                    add_with_carry(a, !b, 1, sf)
                } else {
                    add_with_carry(a, b, 0, sf)
                };
                self.set_sized(rd, sf, result);
                if set_flags {
                    self.set_flags(result, carry, overflow, sf);
                }
                self.pc = next;
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
                let imm = decode_bitmask(sf, n, immr, imms);
                let a = self.get_sized(rn, sf);
                let result = match op {
                    LogicalOp::And => a & imm,
                    LogicalOp::Orr => a | imm,
                    LogicalOp::Eor => a ^ imm,
                    LogicalOp::Ands => a & imm,
                };
                self.set_sized(rd, sf, result);
                if op == LogicalOp::Ands {
                    // A64 sets C from the shifter, and an immediate operand performs no shift.
                    self.set_flags(result, false, false, sf);
                }
                self.pc = next;
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
                let mut b = Self::shift(self.get_sized(rm, sf), shift, amount, sf);
                if invert {
                    b = !b;
                }
                let a = self.get_sized(rn, sf);
                let result = match op {
                    LogicalOp::And => a & b,
                    LogicalOp::Orr => a | b,
                    LogicalOp::Eor => a ^ b,
                    LogicalOp::Ands => a & b,
                };
                self.set_sized(rd, sf, result);
                if op == LogicalOp::Ands {
                    // No shift yields no carry; a shifted operand would carry out of the shifter.
                    self.set_flags(result, amount == 0, false, sf);
                }
                self.pc = next;
            }
            Instr::Bitfield {
                sf,
                signed,
                rd,
                rn,
                immr,
                imms,
            } => {
                let width = if sf { 64 } else { 32 };
                let src = self.get_sized(rn, sf);
                let result = bitfield(src, width, immr, imms, signed);
                self.set_sized(rd, sf, result);
                self.pc = next;
            }
            Instr::Csel {
                sf,
                rd,
                rn,
                rm,
                cond,
                op,
            } => {
                let a = self.get_sized(rn, sf);
                let b = self.get_sized(rm, sf);
                let b = match op {
                    0 => b,
                    1 => b.wrapping_add(1),
                    2 => !b,
                    _ => b.wrapping_neg(),
                };
                let v = if self.cond_holds(cond) { a } else { b };
                self.set_sized(rd, sf, v);
                self.pc = next;
            }
            Instr::MaddSub {
                sf,
                sub,
                rd,
                rn,
                rm,
                ra,
            } => {
                let a = self.get_sized(rn, sf);
                let b = self.get_sized(rm, sf);
                let acc = if ra == 31 {
                    0
                } else {
                    self.get_sized(ra, sf)
                };
                let prod = a.wrapping_mul(b);
                let v = if sub {
                    acc.wrapping_sub(prod)
                } else {
                    acc.wrapping_add(prod)
                };
                self.set_sized(rd, sf, v);
                self.pc = next;
            }
            Instr::ShiftReg {
                sf,
                kind,
                rd,
                rn,
                rm,
            } => {
                let value = self.get_sized(rn, sf);
                // The shift amount is taken modulo the operand width, as A64 specifies.
                let amount = (self.get_sized(rm, sf) & if sf { 63 } else { 31 }) as u8;
                let result = Self::shift(value, kind, amount, sf);
                self.set_sized(rd, sf, result);
                self.pc = next;
            }
            Instr::Div {
                sf,
                signed,
                rd,
                rn,
                rm,
            } => {
                let a = self.get_sized(rn, sf);
                let b = self.get_sized(rm, sf);
                let v = if signed {
                    let (ai, bi) = if sf {
                        (a as i64, b as i64)
                    } else {
                        (a as u32 as i32 as i64, b as u32 as i32 as i64)
                    };
                    // Division by zero yields zero in A64; it does not trap.
                    if bi == 0 { 0 } else { ai.wrapping_div(bi) as u64 }
                } else {
                    a.checked_div(b).unwrap_or(0)
                };
                self.set_sized(rd, sf, v);
                self.pc = next;
            }
            Instr::LoadStore {
                load,
                size,
                signed,
                rt,
                rn,
                mode,
            } => {
                let base = self.get_sp(rn);
                let (address, writeback) = match mode {
                    AddressMode::Offset { imm } | AddressMode::Unscaled { imm } => {
                        (base.wrapping_add_signed(imm), None)
                    }
                    AddressMode::PreIndex { imm } => {
                        let a = base.wrapping_add_signed(imm);
                        (a, Some(a))
                    }
                    AddressMode::PostIndex { imm } => (base, Some(base.wrapping_add_signed(imm))),
                    AddressMode::RegisterOffset {
                        rm,
                        extend,
                        scaled_by_size,
                    } => {
                        let raw = self.get(rm);
                        let extended = match extend {
                            0b010 => u64::from(raw as u32),
                            0b110 => raw as u32 as i32 as i64 as u64,
                            0b011 | 0b111 => raw,
                            _ => return Err(Trap::Unsupported),
                        };
                        let offset = if scaled_by_size {
                            extended.wrapping_shl(size.trailing_zeros())
                        } else {
                            extended
                        };
                        (base.wrapping_add(offset), None)
                    }
                };
                if load {
                    let raw = self.read(address, size)?;
                    let value = if signed {
                        match size {
                            1 => raw as u8 as i8 as i64 as u64,
                            2 => raw as u16 as i16 as i64 as u64,
                            4 => raw as u32 as i32 as i64 as u64,
                            _ => raw,
                        }
                    } else {
                        raw
                    };
                    self.set(rt, value);
                } else {
                    self.write(address, size, self.get(rt))?;
                }
                if let Some(sp) = writeback {
                    self.set_sp(rn, sp);
                }
                self.pc = next;
            }
            Instr::LoadStorePair {
                load,
                sf,
                rt,
                rt2,
                rn,
                mode,
            } => {
                let size = if sf { 8 } else { 4 };
                let base = self.get_sp(rn);
                let (address, writeback) = match mode {
                    AddressMode::Offset { imm } => (base.wrapping_add_signed(imm), None),
                    AddressMode::PreIndex { imm } => {
                        let a = base.wrapping_add_signed(imm);
                        (a, Some(a))
                    }
                    AddressMode::PostIndex { imm } => (base, Some(base.wrapping_add_signed(imm))),
                    _ => return Err(Trap::Unsupported),
                };
                if load {
                    let a = self.read(address, size)?;
                    let b = self.read(address + u64::from(size), size)?;
                    self.set(rt, a);
                    self.set(rt2, b);
                } else {
                    let a = self.get(rt);
                    let b = self.get(rt2);
                    self.write(address, size, a)?;
                    self.write(address + u64::from(size), size, b)?;
                }
                if let Some(sp) = writeback {
                    self.set_sp(rn, sp);
                }
                self.pc = next;
            }
            Instr::Svc { imm } => return Err(Trap::SupervisorCall { imm }),
            Instr::Mrs { rt, .. } => {
                // The interpreter has no system registers; reporting is better than a wrong value.
                let _ = rt;
                return Err(Trap::Unsupported);
            }
            Instr::Msr { .. } => return Err(Trap::Unsupported),
            Instr::FpArith { .. }
            | Instr::FpMov { .. }
            | Instr::FpMovImm { .. }
            | Instr::FpFused { .. }
            | Instr::FpLoadStore { .. } => {
                // Decoded so blocks can be analysed and counted, but there is no FP register file
                // here yet. A wrong answer would be worse than a refusal.
                return Err(Trap::Unsupported);
            }
        }

        Ok(())
    }

    fn set_flags(&mut self, result: u64, carry: bool, overflow: bool, sf: bool) {
        let sign = if sf { 63 } else { 31 };
        self.n = (result >> sign) & 1 != 0;
        self.z = if sf {
            result == 0
        } else {
            result & 0xFFFF_FFFF == 0
        };
        self.c = carry;
        self.v = overflow;
    }

    /// Run until a trap, a stop instruction, or the step limit.
    pub fn run(&mut self, steps: u64) -> Result<u64, Trap> {
        for n in 0..steps {
            self.step()?;
            if self.pc < self.base || self.pc >= self.base + self.memory.len() as u64 {
                return Err(Trap::PcOutOfRange { pc: self.pc });
            }
            let _ = n;
        }
        Err(Trap::StepLimit { steps })
    }
}

/// Add with carry, returning `(result, carry_out, signed_overflow)`, truncated to `sf`.
///
/// `carry_in` of 1 with a complemented operand is how A64 subtraction is defined, and getting that
/// the wrong way round flips every `C` flag and therefore every `HI`/`LS` branch.
fn add_with_carry(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, bool, bool) {
    if sf {
        let (sum, c1) = a.overflowing_add(b);
        let (sum, c2) = sum.overflowing_add(carry_in);
        let carry = c1 || c2;
        let overflow = ((a ^ sum) & (b ^ sum)) >> 63 & 1 != 0;
        (sum, carry, overflow)
    } else {
        let a32 = a as u32;
        let b32 = b as u32;
        let c = carry_in as u32;
        let (s1, c1) = a32.overflowing_add(b32);
        let (s2, c2) = s1.overflowing_add(c);
        let overflow = ((a32 ^ s2) & (b32 ^ s2)) >> 31 & 1 != 0;
        (u64::from(s2), c1 || c2, overflow)
    }
}

/// Decode a logical-immediate bitmask, per the ARM ARM's `DecodeBitMasks`.
///
/// ```text
/// len   = HighestSetBit(immN:~imms)      // a 7-bit field
/// esize = 1 << len
/// S     = imms AND (esize - 1)
/// R     = immr AND (esize - 1)
/// welem = Ones(S + 1)                    // S+1 ones, in an esize-bit element
/// wmask = Replicate(ROR(welem, R))
/// ```
///
/// The first version of this function carried scaffolding that computed nothing; it was replaced
/// with the definition rather than debugged, because a bitmask that is nearly right produces
/// constants that are nearly right, which is the hardest kind of wrong to see.
pub fn decode_bitmask(sf: bool, n: bool, immr: u8, imms: u8) -> u64 {
    let field = (u32::from(n) << 6) | u32::from(!imms & 0x3F);
    // HighestSetBit: index of the most significant one, or "none" when the field is zero.
    if field == 0 {
        return 0;
    }
    let len = 31 - field.leading_zeros();
    let esize = 1u32 << len;
    let levels = esize - 1;
    let s = u32::from(imms) & levels;
    let r = u32::from(immr) & levels;

    let element_mask = if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let welem = if s + 1 >= 64 {
        u64::MAX
    } else {
        (1u64 << (s + 1)) - 1
    } & element_mask;
    let rotated = if r == 0 {
        welem
    } else {
        ((welem >> r) | (welem << (esize - r))) & element_mask
    };

    let width = if sf { 64 } else { 32 };
    let mut out = 0u64;
    let mut shift = 0;
    while shift < width {
        out |= rotated << shift;
        shift += esize;
    }
    if sf { out } else { out & 0xFFFF_FFFF }
}

/// The bitfield operation shared by `UBFM`/`SBFM`, including the `LSL`/`LSR`/`ASR` aliases.
fn bitfield(src: u64, width: u32, immr: u8, imms: u8, signed: bool) -> u64 {
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let src = src & mask;
    if imms >= immr {
        let size = u32::from(imms) - u32::from(immr) + 1;
        let shifted = (src >> u32::from(immr)) & if size >= 64 { u64::MAX } else { (1u64 << size) - 1 };
        if signed {
            // SBFM sign-extends from bit `size-1` when imms < width-1.
            sign_extend_to(shifted, size, width)
        } else {
            shifted & mask
        }
    } else {
        // Rotate right by immr, then take the top imms+1 bits.
        let rotated = if width >= 64 {
            src.rotate_right(u32::from(immr))
        } else {
            (src as u32).rotate_right(u32::from(immr) % width) as u64
        };
        let size = u32::from(imms) + 1;
        let out = rotated & if size >= 64 { u64::MAX } else { (1u64 << size) - 1 };
        if signed {
            sign_extend_to(out, size, width)
        } else {
            out & mask
        }
    }
}

fn sign_extend_to(value: u64, from: u32, to: u32) -> u64 {
    if from == 0 || from >= to {
        return value;
    }
    let sign = (value >> (from - 1)) & 1 != 0;
    let mask = (1u64 << from) - 1;
    let v = value & mask;
    if sign {
        v | !mask
    } else {
        v
    }
}
