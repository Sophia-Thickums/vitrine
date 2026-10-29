//! AArch64 (ARMv8.2-A) instruction decoder.
//!
//! The front end of a JIT: turn a 32-bit word into a typed instruction. This module is the
//! foundation the recompiler sits on, and it is written in Rust for one measurable reason — a
//! decoder is a thousand small bit-field decisions, and in C++ a wrong mask reads as a valid
//! instruction and produces wrong code three functions later. Here a mask that matches nothing
//! falls through to [`Instr::Unknown`], which the caller can refuse to compile.
//!
//! # Scope
//!
//! Deliberately limited to what a block recompiler needs to walk code and decide control flow,
//! plus the arithmetic and memory instructions that dominate real Switch binaries:
//!
//! - control flow: `B`, `BL`, `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`, `BR`, `BLR`, `RET`
//! - addresses: `ADR`, `ADRP`
//! - immediates: `MOVN`/`MOVZ`/`MOVK`
//! - integer arithmetic: add/sub immediate and shifted-register, logical immediate and
//!   shifted-register, `UBFM`/`SBFM` (the `LSL`/`LSR`/`ASR` aliases)
//! - memory: load/store unsigned-offset, pre/post-index and register-offset, plus load/store pair
//! - system: `SVC`, `MRS`, `MSR`, `NOP`, `HINT`
//! - a small floating-point set: add/sub/mul/div and `FMOV` register-to-register
//!
//! Everything else decodes to [`Instr::Unknown`] with the original word preserved, so an
//! unimplemented instruction is visible in a log rather than silently mis-executed.
//!
//! # Verification
//!
//! Not from memory. `llvm-mc -triple=aarch64` produces the encodings, and `tests/a64.rs` asserts
//! this decoder agrees with it instruction by instruction. When they disagree, the assembler is
//! right and this file is wrong.

use std::fmt;

/// Condition codes, as encoded in the low four bits of a conditional instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

impl Cond {
    fn from_bits(bits: u32) -> Self {
        use Cond::*;
        match bits & 0xF {
            0 => Eq,
            1 => Ne,
            2 => Cs,
            3 => Cc,
            4 => Mi,
            5 => Pl,
            6 => Vs,
            7 => Vc,
            8 => Hi,
            9 => Ls,
            10 => Ge,
            11 => Lt,
            12 => Gt,
            13 => Le,
            14 => Al,
            _ => Nv,
        }
    }

    /// The condition with its sense inverted; `AL` and `NV` are their own inverses' fixed points.
    pub fn inverted(self) -> Self {
        use Cond::*;
        match self {
            Eq => Ne,
            Ne => Eq,
            Cs => Cc,
            Cc => Cs,
            Mi => Pl,
            Pl => Mi,
            Vs => Vc,
            Vc => Vs,
            Hi => Ls,
            Ls => Hi,
            Ge => Lt,
            Lt => Ge,
            Gt => Le,
            Le => Gt,
            Al => Nv,
            Nv => Al,
        }
    }
}

impl fmt::Display for Cond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Cs => "cs",
            Cond::Cc => "cc",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
            Cond::Nv => "nv",
        };
        f.write_str(s)
    }
}

/// How a shifted-register operand is shifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

impl ShiftType {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Lsl,
            1 => Self::Lsr,
            2 => Self::Asr,
            _ => Self::Ror,
        }
    }
}

/// A decoded instruction.
///
/// `sf` is true for 64-bit operand size, false for 32-bit — the single most common source of
/// silent bugs in a decoder, so it is a named field on every instruction that has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instr {
    /// `NOP`.
    Nop,
    /// `RET {Xn}`.
    Ret { rn: u8 },
    /// `BR Xn`.
    Br { rn: u8 },
    /// `BLR Xn`.
    Blr { rn: u8 },
    /// `B <offset>` (byte offset, signed).
    B { offset: i64 },
    /// `BL <offset>` (byte offset, signed).
    Bl { offset: i64 },
    /// `B.<cond> <offset>`.
    BCond { cond: Cond, offset: i64 },
    /// `CBZ`/`CBNZ`: `nonzero` distinguishes them.
    Cbz {
        sf: bool,
        nonzero: bool,
        rt: u8,
        offset: i64,
    },
    /// `TBZ`/`TBNZ`.
    Tbz {
        nonzero: bool,
        rt: u8,
        bit: u8,
        offset: i64,
    },
    /// `ADR Xd, <offset>` — offset relative to the instruction.
    Adr { rd: u8, offset: i64 },
    /// `ADRP Xd, <page>` — page offset in bytes, relative to the instruction's page.
    Adrp { rd: u8, page_offset: i64 },
    /// `MOVN`/`MOVZ`/`MOVK`.
    Mov {
        sf: bool,
        op: MovOp,
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    /// Add/subtract with a 12-bit immediate, optionally shifted left by 12.
    AddSubImm {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        imm: u32,
        lsl12: bool,
    },
    /// Add/subtract with a shifted register.
    AddSubShifted {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    /// `AND`/`ORR`/`EOR`/`ANDS` with a bitmask immediate.
    LogicalImm {
        sf: bool,
        op: LogicalOp,
        rd: u8,
        rn: u8,
        n: bool,
        immr: u8,
        imms: u8,
    },
    /// `AND`/`ORR`/`EOR`/`BIC`/`ORN`/`EON`/`ANDS` with a shifted register.
    LogicalShifted {
        sf: bool,
        op: LogicalOp,
        invert: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    /// `UBFM`/`SBFM` — the implementer of `LSL`, `LSR`, `ASR`, `UXTB` and friends.
    Bitfield {
        sf: bool,
        signed: bool,
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    /// `CSEL`/`CSINC`/`CSINV`/`CSNEG`.
    Csel {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
        op: u8,
    },
    /// `MADD`/`MSUB` (and the long forms).
    MaddSub {
        sf: bool,
        sub: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `LSLV`/`LSRV`/`ASRV`/`RORV` — shift by a register, not an immediate.
    ShiftReg {
        sf: bool,
        kind: ShiftType,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `UDIV`/`SDIV`.
    Div {
        sf: bool,
        signed: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// A load or store: scalar, at some address mode.
    LoadStore {
        /// True for a load.
        load: bool,
        /// Access size in bytes (1, 2, 4, 8).
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddressMode,
        /// Sign extension applied on load (LDRSB/LDRSH/LDRSW).
        signed: bool,
    },
    /// A load or store of a register pair (`LDP`/`STP`).
    LoadStorePair {
        load: bool,
        sf: bool,
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddressMode,
    },
    /// `SVC #imm` — a supervisor call. The HLE layer's entry point.
    Svc { imm: u16 },
    /// `MRS Xt, <system reg>`.
    Mrs { rt: u8, sysreg: u32 },
    /// `MSR <system reg>, Xt`.
    Msr { sysreg: u32, rt: u8 },
    /// Floating-point add/sub/mul/div, single or double precision.
    FpArith {
        dp: bool,
        op: FpOp,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `FMOV Dd, Dn` / `FMOV Sd, Sn` — register to register.
    FpMov { dp: bool, rd: u8, rn: u8 },
    /// `FMOV Dd, #imm` / `FMOV Sd, #imm` — a floating-point immediate (the 8-bit form).
    FpMovImm { dp: bool, rd: u8, imm8: u8 },
    /// `FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`.
    FpFused {
        dp: bool,
        op: u8,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `FMOV Xd, Dn` — FP register to general register, bit-for-bit.
    FpToGen { dp: bool, rd: u8, rn: u8 },
    /// `FMOV Dd, Xn` — general register to FP register, bit-for-bit.
    GenToFp { dp: bool, rd: u8, rn: u8 },
    /// A scalar floating-point load or store (`LDR D0, [X1]` and friends).
    ///
    /// Decoded so that a block can be *analysed* and counted; executing it needs an FP register
    /// file, which this crate does not have yet.
    FpLoadStore {
        load: bool,
        /// Access size in bytes (1, 2, 4, 8, 16).
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddressMode,
    },
    /// An instruction this decoder does not implement. The word is preserved so a caller can
    /// log it and refuse to compile the block.
    Unknown { word: u32 },
}

/// Which `MOV` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovOp {
    /// `MOVN` — move wide with NOT.
    Not,
    /// `MOVZ` — move wide with zero.
    Zero,
    /// `MOVK` — move wide with keep.
    Keep,
}

/// Which logical operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Orr,
    Eor,
    Ands,
}

impl LogicalOp {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::And,
            1 => Self::Orr,
            2 => Self::Eor,
            _ => Self::Ands,
        }
    }
}

/// Floating-point operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

/// How the address of a load/store is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    /// `[Xn, #imm]` — unsigned offset, base unchanged.
    Offset { imm: i64 },
    /// `[Xn, #imm]` — unscaled offset (`LDUR`/`STUR`), base unchanged.
    Unscaled { imm: i64 },
    /// `[Xn, #imm]!` — pre-index, base updated.
    PreIndex { imm: i64 },
    /// `[Xn], #imm` — post-index, base updated.
    PostIndex { imm: i64 },
    /// `[Xn, Wm/Xm]` — register offset, with the extend option and (only for `LSL`) a shift.
    ///
    /// `extend` is the raw 3-bit `option` field: 010 UXTW, 011 LSL, 110 SXTW, 111 SXTX.
    /// `amount` carries the `S` bit: when set, the extended value is scaled by the access size.
    /// **Only `LSL` is a shift**; for `UXTW`/`SXTW`/`SXTX` the register is extended to 64 bits and
    /// then scaled, which is why this is a bool rather than a `ShiftType`.
    RegisterOffset {
        rm: u8,
        extend: u8,
        scaled_by_size: bool,
    },
}

/// Sign-extend the low `bits` of `value` and shift the result left by `shift`.
///
/// The single most-used helper in the file: every branch and address offset goes through it, and
/// getting the width wrong produces jumps to plausible-looking wrong addresses.
pub fn sign_extend_shifted(value: u32, bits: u32, shift: u32) -> i64 {
    let shift_amount = 64 - bits;
    let v = (i64::from(value as i32) << shift_amount) >> shift_amount;
    v << shift
}

/// Sign-extend a value with `bits` significant bits, without shifting.
pub fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

impl Instr {
    /// True when the instruction ends a basic block (unconditional transfer or return).
    pub fn ends_block(&self) -> bool {
        matches!(
            self,
            Instr::B { .. } | Instr::Br { .. } | Instr::Ret { .. } | Instr::Blr { .. }
        )
    }

    /// True when the instruction is a call (returns to the next instruction).
    pub fn is_call(&self) -> bool {
        matches!(self, Instr::Bl { .. } | Instr::Blr { .. })
    }
}

/// Decode one 32-bit AArch64 instruction word.
pub fn decode(word: u32) -> Instr {
    // Branch and system first: they share top bits with nothing else and a wrong mask there
    // would reinterpret a `B` as arithmetic.
    if let Some(i) = decode_branch(word) {
        return i;
    }
    if let Some(i) = decode_system(word) {
        return i;
    }
    if let Some(i) = decode_adr(word) {
        return i;
    }
    if let Some(i) = decode_mov_wide(word) {
        return i;
    }
    if let Some(i) = decode_add_sub_imm(word) {
        return i;
    }
    if let Some(i) = decode_bitfield(word) {
        return i;
    }
    if let Some(i) = decode_logical_imm(word) {
        return i;
    }
    if let Some(i) = decode_add_sub_shifted(word) {
        return i;
    }
    if let Some(i) = decode_logical_shifted(word) {
        return i;
    }
    if let Some(i) = decode_cond_select(word) {
        return i;
    }
    if let Some(i) = decode_three_source(word) {
        return i;
    }
    if let Some(i) = decode_load_store(word) {
        return i;
    }
    if let Some(i) = decode_fp(word) {
        return i;
    }
    Instr::Unknown { word }
}

fn decode_branch(word: u32) -> Option<Instr> {
    // B / BL: bits 30-26 = 00101, op bit 31 selects BL.
    if word & 0x7C00_0000 == 0x1400_0000 {
        let offset = sign_extend_shifted(word & 0x03FF_FFFF, 26, 2);
        return Some(if word & 0x8000_0000 != 0 {
            Instr::Bl { offset }
        } else {
            Instr::B { offset }
        });
    }
    // B.cond: 0101 0100 imm19 0 cond
    if word & 0xFF00_0010 == 0x5400_0000 {
        return Some(Instr::BCond {
            cond: Cond::from_bits(word),
            offset: sign_extend_shifted((word >> 5) & 0x7_FFFF, 19, 2),
        });
    }
    // CBZ / CBNZ: sf 011010 0|1 imm19 Rt
    if word & 0x7E00_0000 == 0x3400_0000 {
        let sf = word >> 31 != 0;
        let nonzero = (word >> 24) & 1 != 0;
        // A 32-bit form must have bit 31 clear; bit 31 set means a 64-bit form. Anything else
        // (bit 30..29 patterns) is filtered by the mask above.
        return Some(Instr::Cbz {
            sf,
            nonzero,
            rt: (word & 0x1F) as u8,
            offset: sign_extend_shifted((word >> 5) & 0x7_FFFF, 19, 2),
        });
    }
    // TBZ / TBNZ: b5 011011 0|1 b40 imm14 Rt
    if word & 0x7E00_0000 == 0x3600_0000 {
        let bit5 = (word >> 31) & 1;
        let bit40 = (word >> 19) & 0x1F;
        return Some(Instr::Tbz {
            nonzero: (word >> 24) & 1 != 0,
            rt: (word & 0x1F) as u8,
            bit: ((bit5 << 5) | bit40) as u8,
            offset: sign_extend_shifted((word >> 5) & 0x3FFF, 14, 2),
        });
    }
    // Unconditional branch register: 1101011 opc 11111 000000 Rn 00000
    if word & 0xFE00_0000 == 0xD600_0000 {
        let opc = (word >> 21) & 0xF;
        let rn = ((word >> 5) & 0x1F) as u8;
        return Some(match opc {
            0 => Instr::Br { rn },
            1 => Instr::Blr { rn },
            2 => Instr::Ret { rn },
            _ => Instr::Unknown { word },
        });
    }
    None
}

fn decode_system(word: u32) -> Option<Instr> {
    if word == 0xD503_201F {
        return Some(Instr::Nop);
    }
    // HINT #imm: 1101 0101 0000 0011 0010 0000 0001 1111 with imm in bits 11-5. Emit as Nop for
    // the ones that are architecturally no-ops (YIELD/WFE/WFI are not, so they stay Unknown).
    if word & 0xFFFF_F01F == 0xD503_201F {
        let imm = (word >> 5) & 0x7F;
        if imm == 0 {
            return Some(Instr::Nop);
        }
        return Some(Instr::Unknown { word });
    }
    // SVC: 1101 0100 000 imm16 000 01
    if word & 0xFFE0_001F == 0xD400_0001 {
        return Some(Instr::Svc {
            imm: ((word >> 5) & 0xFFFF) as u16,
        });
    }
    // MRS/MSR (register): 1101 0101 00 L 1 <sysreg> Rt
    if word & 0xFFD0_0000 == 0xD510_0000 {
        let read = (word >> 21) & 1 != 0;
        let sysreg = (word >> 5) & 0xFFFF;
        let rt = (word & 0x1F) as u8;
        return Some(if read {
            Instr::Mrs { rt, sysreg }
        } else {
            Instr::Msr { sysreg, rt }
        });
    }
    None
}

/// `ADR`/`ADRP`: `op immlo(2) 10000 immhi(19) Rd`.
///
/// ADRP's immediate is a *page number* — the 21-bit field shifted left 12 — not a byte offset.
/// Getting that wrong produces an address that looks plausible and is off by a factor of 4096.
fn decode_adr(word: u32) -> Option<Instr> {
    if word & 0x1F00_0000 != 0x1000_0000 {
        return None;
    }
    let immlo = (word >> 29) & 3;
    let immhi = (word >> 5) & 0x7_FFFF;
    let imm = sign_extend(u64::from((immhi << 2) | immlo), 21);
    let rd = (word & 0x1F) as u8;
    Some(if word >> 31 != 0 {
        Instr::Adrp {
            rd,
            page_offset: imm << 12,
        }
    } else {
        Instr::Adr { rd, offset: imm }
    })
}

fn decode_mov_wide(word: u32) -> Option<Instr> {
    // sf 10 opc(2) 100101 hw(2) imm16 Rd
    if word & 0x1F80_0000 != 0x1280_0000 {
        return None;
    }
    let sf = word >> 31 != 0;
    let op = match (word >> 29) & 3 {
        0 => MovOp::Not,
        2 => MovOp::Zero,
        3 => MovOp::Keep,
        _ => return None,
    };
    let shift = ((word >> 21) & 3) * 16;
    if !sf && shift == 48 {
        return Some(Instr::Unknown { word });
    }
    Some(Instr::Mov {
        sf,
        op,
        rd: (word & 0x1F) as u8,
        imm16: ((word >> 5) & 0xFFFF) as u16,
        shift: shift as u8,
    })
}

fn decode_add_sub_imm(word: u32) -> Option<Instr> {
    // sf op S 100010 sh imm12 Rn Rd
    if word & 0x1F00_0000 != 0x1100_0000 {
        return None;
    }
    Some(Instr::AddSubImm {
        sf: word >> 31 != 0,
        sub: (word >> 30) & 1 != 0,
        set_flags: (word >> 29) & 1 != 0,
        lsl12: (word >> 22) & 1 != 0,
        imm: (word >> 10) & 0xFFF,
        rn: ((word >> 5) & 0x1F) as u8,
        rd: (word & 0x1F) as u8,
    })
}

fn decode_bitfield(word: u32) -> Option<Instr> {
    // sf opc(2) 100110 N immr imms Rn Rd — UBFM is opc=10, SBFM is opc=00.
    if word & 0x1F80_0000 != 0x1300_0000 {
        return None;
    }
    let opc = (word >> 29) & 3;
    let signed = match opc {
        0 => true,                                 // SBFM
        2 => false,                                // UBFM
        _ => return Some(Instr::Unknown { word }), // BFM and reserved
    };
    let sf = word >> 31 != 0;
    let n = (word >> 22) & 1 != 0;
    // N must equal sf for a valid bitfield instruction.
    if n != sf {
        return Some(Instr::Unknown { word });
    }
    Some(Instr::Bitfield {
        sf,
        signed,
        rd: (word & 0x1F) as u8,
        rn: ((word >> 5) & 0x1F) as u8,
        immr: ((word >> 16) & 0x3F) as u8,
        imms: ((word >> 10) & 0x3F) as u8,
    })
}

fn decode_logical_imm(word: u32) -> Option<Instr> {
    // sf opc(2) 100100 N immr imms Rn Rd
    if word & 0x1F80_0000 != 0x1200_0000 {
        return None;
    }
    let sf = word >> 31 != 0;
    let op = LogicalOp::from_bits((word >> 29) & 3);
    let n = (word >> 22) & 1 != 0;
    if n != sf {
        return Some(Instr::Unknown { word });
    }
    Some(Instr::LogicalImm {
        sf,
        op,
        n,
        rd: (word & 0x1F) as u8,
        rn: ((word >> 5) & 0x1F) as u8,
        immr: ((word >> 16) & 0x3F) as u8,
        imms: ((word >> 10) & 0x3F) as u8,
    })
}

fn decode_add_sub_shifted(word: u32) -> Option<Instr> {
    // sf op S 01011 shift(2) 0 Rm imm6 Rn Rd
    if word & 0x1F20_0000 != 0x0B00_0000 {
        return None;
    }
    let sf = word >> 31 != 0;
    let amount = ((word >> 10) & 0x3F) as u8;
    if !sf && amount >= 32 {
        return Some(Instr::Unknown { word });
    }
    Some(Instr::AddSubShifted {
        sf,
        sub: (word >> 30) & 1 != 0,
        set_flags: (word >> 29) & 1 != 0,
        shift: ShiftType::from_bits((word >> 22) & 3),
        rm: ((word >> 16) & 0x1F) as u8,
        amount,
        rn: ((word >> 5) & 0x1F) as u8,
        rd: (word & 0x1F) as u8,
    })
}

fn decode_logical_shifted(word: u32) -> Option<Instr> {
    // sf opc(2) 01010 shift(2) N Rm imm6 Rn Rd
    if word & 0x1F00_0000 != 0x0A00_0000 {
        return None;
    }
    let sf = word >> 31 != 0;
    let amount = ((word >> 10) & 0x3F) as u8;
    let n = (word >> 21) & 1 != 0;
    let op_bits = (word >> 29) & 3;
    if op_bits == 3 {
        // ANDS is legal; opc 3 with N=1 set is BIC/ORN/EON, represented by `invert`.
        return Some(Instr::LogicalShifted {
            sf,
            op: LogicalOp::Ands,
            invert: n,
            rd: (word & 0x1F) as u8,
            rn: ((word >> 5) & 0x1F) as u8,
            rm: ((word >> 16) & 0x1F) as u8,
            shift: ShiftType::from_bits((word >> 22) & 3),
            amount,
        });
    }
    Some(Instr::LogicalShifted {
        sf,
        op: LogicalOp::from_bits(op_bits),
        invert: n,
        rd: (word & 0x1F) as u8,
        rn: ((word >> 5) & 0x1F) as u8,
        rm: ((word >> 16) & 0x1F) as u8,
        shift: ShiftType::from_bits((word >> 22) & 3),
        amount,
    })
}

fn decode_cond_select(word: u32) -> Option<Instr> {
    // sf op 11010100 Rm cond op2 Rn Rd
    if word & 0x1FE0_0000 != 0x1A80_0000 {
        return None;
    }
    let op2 = (word >> 10) & 3;
    if op2 > 1 {
        return None;
    }
    Some(Instr::Csel {
        sf: word >> 31 != 0,
        op: (((word >> 30) & 1) << 1 | op2) as u8,
        rm: ((word >> 16) & 0x1F) as u8,
        cond: Cond::from_bits((word >> 12) & 0xF),
        rn: ((word >> 5) & 0x1F) as u8,
        rd: (word & 0x1F) as u8,
    })
}

fn decode_three_source(word: u32) -> Option<Instr> {
    // sf 00 11011 op31 Rm o0 Ra Rn Rd — MADD/MSUB when op31 == 0.
    if word & 0x1F00_0000 == 0x1B00_0000 {
        let op31 = (word >> 21) & 7;
        let o0 = (word >> 15) & 1;
        if op31 == 0 {
            return Some(Instr::MaddSub {
                sf: word >> 31 != 0,
                sub: o0 != 0,
                rm: ((word >> 16) & 0x1F) as u8,
                ra: ((word >> 10) & 0x1F) as u8,
                rn: ((word >> 5) & 0x1F) as u8,
                rd: (word & 0x1F) as u8,
            });
        }
        return Some(Instr::Unknown { word });
    }
    // UDIV/SDIV: sf 00 11010110 Rm opcode(6) Rn Rd, opcode 000010 unsigned, 000011 signed.
    if word & 0x1FE0_0000 == 0x1AC0_0000 {
        let opcode = (word >> 10) & 0x3F;
        // The same encoding space holds the register-controlled shifts.
        if (0b001000..=0b001011).contains(&opcode) {
            let kind = match opcode {
                0b001000 => ShiftType::Lsl,
                0b001001 => ShiftType::Lsr,
                0b001010 => ShiftType::Asr,
                _ => ShiftType::Ror,
            };
            return Some(Instr::ShiftReg {
                sf: word >> 31 != 0,
                kind,
                rm: ((word >> 16) & 0x1F) as u8,
                rn: ((word >> 5) & 0x1F) as u8,
                rd: (word & 0x1F) as u8,
            });
        }
        let signed = match opcode {
            0b000010 => false,
            0b000011 => true,
            _ => return Some(Instr::Unknown { word }),
        };
        return Some(Instr::Div {
            sf: word >> 31 != 0,
            signed,
            rm: ((word >> 16) & 0x1F) as u8,
            rn: ((word >> 5) & 0x1F) as u8,
            rd: (word & 0x1F) as u8,
        });
    }
    None
}

fn decode_load_store(word: u32) -> Option<Instr> {
    // Load/store pair: opc(2) 101 V 0 1|0 L imm7 Rt2 Rn Rt
    if word & 0x3A00_0000 == 0x2800_0000 {
        let v = (word >> 26) & 1 != 0;
        if v {
            return Some(Instr::Unknown { word });
        }
        let opc = (word >> 30) & 3;
        let sf = opc & 2 != 0;
        let load = (word >> 22) & 1 != 0;
        let scale = if sf { 8 } else { 4 };
        let imm7 = sign_extend(u64::from((word >> 15) & 0x7F), 7) * scale;
        let mode = match (word >> 23) & 3 {
            1 => AddressMode::PostIndex { imm: imm7 },
            2 => AddressMode::Offset { imm: imm7 },
            3 => AddressMode::PreIndex { imm: imm7 },
            _ => return Some(Instr::Unknown { word }),
        };
        return Some(Instr::LoadStorePair {
            load,
            sf,
            rt: (word & 0x1F) as u8,
            rt2: ((word >> 10) & 0x1F) as u8,
            rn: ((word >> 5) & 0x1F) as u8,
            mode,
        });
    }

    // Single-register load/store: size(2) 111 V 00 .. The discriminator is bits 29-27 == 111 with
    // V (bit 26) clear; the addressing mode then lives in bits 25-24, and the opcode in 23-22.
    //
    // An earlier version masked bits 25-24 as well, which matched only the unsigned-offset group
    // and threw away LDUR/STUR and every register-offset access.
    if (word >> 27) & 0x7 != 0x7 {
        return None;
    }
    if (word >> 26) & 1 != 0 {
        // SIMD/FP load/store. The scalar forms (LDR/STR of B/H/S/D/Q) share this space; the
        // vector-structure forms (LD1/ST2/...) do not, and are reported rather than guessed.
        let opc = (word >> 22) & 3;
        if opc == 3 {
            return Some(Instr::Unknown { word });
        }
        let load = opc & 1 != 0;
        let size = 1u8 << ((word >> 30) & 3);
        let rt = (word & 0x1F) as u8;
        let rn = ((word >> 5) & 0x1F) as u8;
        let mode = match (word >> 24) & 3 {
            1 => AddressMode::Offset {
                imm: i64::from((word >> 10) & 0xFFF) * i64::from(size),
            },
            0 => {
                if (word >> 21) & 1 != 0 {
                    AddressMode::RegisterOffset {
                        rm: ((word >> 16) & 0x1F) as u8,
                        extend: ((word >> 13) & 7) as u8,
                        scaled_by_size: (word >> 12) & 1 != 0,
                    }
                } else {
                    let imm9 = sign_extend(u64::from((word >> 12) & 0x1FF), 9);
                    match (word >> 10) & 3 {
                        0 => AddressMode::Unscaled { imm: imm9 },
                        1 => AddressMode::PostIndex { imm: imm9 },
                        3 => AddressMode::PreIndex { imm: imm9 },
                        _ => return Some(Instr::Unknown { word }),
                    }
                }
            }
            _ => return Some(Instr::Unknown { word }),
        };
        return Some(Instr::FpLoadStore {
            load,
            size,
            rt,
            rn,
            mode,
        });
    }

    let size_bits = (word >> 30) & 3;
    let opc = (word >> 22) & 3;
    let size = 1u8 << size_bits;
    let rt = (word & 0x1F) as u8;
    let rn = ((word >> 5) & 0x1F) as u8;

    // opc: 00 store, 01 load zero-extended, 10 sign-extend into 64 bits, 11 sign-extend to the
    // access width. `opc == 10` is *not* only LDRSW: for a byte or halfword it is LDRSB/LDRSH into
    // an X register. The only opc-10 encoding that is not a load is PRFM, which shares it at
    // size == 11; that one is reported rather than decoded as a load of register `prfop`.
    if opc == 0b10 && size_bits == 0b11 {
        return Some(Instr::Unknown { word });
    }
    let load = opc != 0b00;
    let signed = opc & 0b10 != 0;

    let mode = match (word >> 24) & 3 {
        // 00: unscaled, post-index, pre-index, or register offset (bit 21 selects the last).
        0 => {
            if (word >> 21) & 1 != 0 {
                AddressMode::RegisterOffset {
                    rm: ((word >> 16) & 0x1F) as u8,
                    extend: ((word >> 13) & 7) as u8,
                    scaled_by_size: (word >> 12) & 1 != 0,
                }
            } else {
                let imm9 = sign_extend(u64::from((word >> 12) & 0x1FF), 9);
                match (word >> 10) & 3 {
                    0 => AddressMode::Unscaled { imm: imm9 },
                    1 => AddressMode::PostIndex { imm: imm9 },
                    3 => AddressMode::PreIndex { imm: imm9 },
                    _ => return Some(Instr::Unknown { word }),
                }
            }
        }
        // 01: unsigned offset, scaled by the access size.
        1 => AddressMode::Offset {
            imm: i64::from((word >> 10) & 0xFFF) * i64::from(size),
        },
        // 10: unprivileged (LDTR/STTR); 11: reserved in this group.
        _ => return Some(Instr::Unknown { word }),
    };

    Some(Instr::LoadStore {
        load,
        size,
        signed,
        rt,
        rn,
        mode,
    })
}

fn decode_fp(word: u32) -> Option<Instr> {
    // FMADD/FMSUB/FNMADD/FNMSUB: M 0 S 11111 type 0 Rm 0 o1 Ra Rn Rd
    if word & 0x5F00_0000 == 0x1F00_0000 {
        let dp = (word >> 22) & 1 != 0;
        return Some(Instr::FpFused {
            dp,
            op: (((word >> 21) & 1) << 1 | ((word >> 15) & 1)) as u8,
            rm: ((word >> 16) & 0x1F) as u8,
            ra: ((word >> 10) & 0x1F) as u8,
            rn: ((word >> 5) & 0x1F) as u8,
            rd: (word & 0x1F) as u8,
        });
    }
    // FMOV (scalar, immediate): sf 0 0 11110 type 1 imm8 100 00000 Rd
    if word & 0xFFB0_1FE0 == 0x1E20_1000 {
        return Some(Instr::FpMovImm {
            dp: (word >> 22) & 1 != 0,
            imm8: ((word >> 13) & 0xFF) as u8,
            rd: (word & 0x1F) as u8,
        });
    }
    // FMOV (general): 1 0 0 11110 type 1 rmode opcode 000000 Rn Rd
    //   rmode 00, opcode 110/111: FMOV Wd/Xd <- Sn/Dn (to general)
    //   rmode 01, opcode 110/111: FMOV Sd/Dd <- Wn/Xn (from general)
    if word & 0x7F3F_FC00 == 0x1E26_0000 {
        let dp = (word >> 22) & 1 != 0;
        let rmode = (word >> 19) & 3;
        let opcode = (word >> 16) & 0x1F;
        let rn = ((word >> 5) & 0x1F) as u8;
        let rd = (word & 0x1F) as u8;
        return Some(match (rmode, opcode) {
            (0, 0b110) | (0, 0b111) => Instr::FpToGen { dp, rd, rn },
            (1, 0b110) | (1, 0b111) => Instr::GenToFp { dp, rd, rn },
            _ => Instr::Unknown { word },
        });
    }
    // FMOV register: sf 000 11110 1 type 1 000000 10000 Rn Rd
    if word & 0xFFBF_FC00 == 0x1E20_4000 {
        let dp = (word >> 22) & 1 != 0;
        return Some(Instr::FpMov {
            dp,
            rn: ((word >> 5) & 0x1F) as u8,
            rd: (word & 0x1F) as u8,
        });
    }
    // FP data processing (2 source/3 source): M 0 S 11110 type 1 Rm opcode(6) Rn Rd
    if word & 0x5F00_0000 == 0x1E00_0000 {
        let dp = (word >> 22) & 1 != 0;
        let rm = ((word >> 16) & 0x1F) as u8;
        let rn = ((word >> 5) & 0x1F) as u8;
        let rd = (word & 0x1F) as u8;
        let opcode = (word >> 10) & 0x3F;
        let op = match opcode {
            0b000010 => FpOp::Mul,
            0b000110 => FpOp::Div,
            0b001010 => FpOp::Add,
            0b001110 => FpOp::Sub,
            _ => return Some(Instr::Unknown { word }),
        };
        return Some(Instr::FpArith { dp, op, rd, rn, rm });
    }
    None
}

impl Instr {
    /// The mnemonic, matching the spelling `llvm-mc` prints, including the aliases it prefers.
    ///
    /// `llvm-mc` does not print `UBFM` for an `lsl`; it re-aliases on the way out, and a decoder
    /// that disagrees with the assembler here is not necessarily wrong but cannot be diffed
    /// against it. So the alias rules are implemented rather than skipped: a `UBFM` whose field
    /// values spell `LSL` prints `lsl`, an `ANDS` to `XZR` prints `tst`, and so on.
    pub fn mnemonic(&self) -> String {
        match *self {
            Instr::Nop => "nop".into(),
            Instr::Ret { .. } => "ret".into(),
            Instr::Br { .. } => "br".into(),
            Instr::Blr { .. } => "blr".into(),
            Instr::B { .. } => "b".into(),
            Instr::Bl { .. } => "bl".into(),
            Instr::BCond { cond, .. } => format!("b.{cond}"),
            Instr::Cbz { nonzero, .. } => if nonzero { "cbnz" } else { "cbz" }.into(),
            Instr::Tbz { nonzero, .. } => if nonzero { "tbnz" } else { "tbz" }.into(),
            Instr::Adr { .. } => "adr".into(),
            Instr::Adrp { .. } => "adrp".into(),
            Instr::Mov { op, shift, .. } => match op {
                // MOVZ/MOVN with no shift are printed as `mov`.
                MovOp::Zero | MovOp::Not if shift == 0 => "mov".into(),
                MovOp::Zero => "movz".into(),
                MovOp::Not => "movn".into(),
                MovOp::Keep => "movk".into(),
            },
            Instr::AddSubImm {
                set_flags,
                sub,
                rd,
                lsl12,
                imm,
                ..
            } => match (set_flags, sub) {
                (false, false) if rd == 31 => "mov".into(), // ADD XZR? not an alias; keep add
                (false, false) => "add".into(),
                (false, true) => "sub".into(),
                (true, true) if rd == 31 && !lsl12 => "cmp".into(),
                (true, false) if rd == 31 && !lsl12 && imm == 0 => "cmn".into(),
                (true, false) => "adds".into(),
                (true, true) => "subs".into(),
            },
            Instr::AddSubShifted {
                set_flags,
                sub,
                rd,
                shift,
                amount,
                ..
            } => match (set_flags, sub) {
                (true, true) if rd == 31 && shift == ShiftType::Lsl && amount == 0 => "cmp".into(),
                (true, false) if rd == 31 && shift == ShiftType::Lsl && amount == 0 => "cmn".into(),
                (false, false) => "add".into(),
                (false, true) => "sub".into(),
                (true, false) => "adds".into(),
                (true, true) => "subs".into(),
            },
            Instr::LogicalImm { op, rd, .. } => match op {
                LogicalOp::Ands if rd == 31 => "tst".into(),
                LogicalOp::And => "and".into(),
                LogicalOp::Orr => "orr".into(),
                LogicalOp::Eor => "eor".into(),
                LogicalOp::Ands => "ands".into(),
            },
            Instr::LogicalShifted {
                op,
                invert,
                rd,
                rn,
                shift,
                amount,
                ..
            } => {
                // ORR Xd, XZR, Xm is printed as `mov Xd, Xm`.
                if op == LogicalOp::Orr
                    && !invert
                    && rn == 31
                    && shift == ShiftType::Lsl
                    && amount == 0
                {
                    return "mov".into();
                }
                if op == LogicalOp::Ands && rd == 31 && !invert {
                    return "tst".into();
                }
                match (op, invert) {
                    (LogicalOp::And, false) => "and".into(),
                    (LogicalOp::Orr, false) => "orr".into(),
                    (LogicalOp::Eor, false) => "eor".into(),
                    (LogicalOp::Ands, false) => "ands".into(),
                    (LogicalOp::And, true) => "bic".into(),
                    (LogicalOp::Orr, true) => "orn".into(),
                    (LogicalOp::Eor, true) => "eon".into(),
                    (LogicalOp::Ands, true) => "bics".into(),
                }
            }
            Instr::Bitfield {
                sf,
                signed,
                immr,
                imms,
                ..
            } => {
                let width = if sf { 63 } else { 31 };
                let (immr, imms) = (u32::from(immr), u32::from(imms));
                if !signed {
                    // UBFM immr=width-shift, imms=width-1-shift  =>  imms == immr - 1.
                    if immr != 0 && imms == immr - 1 {
                        "lsl".into()
                    } else if imms == width {
                        "lsr".into()
                    } else {
                        "ubfm".into()
                    }
                } else if imms == width {
                    "asr".into()
                } else {
                    "sbfm".into()
                }
            }
            Instr::Csel { .. } => "csel".into(),
            Instr::MaddSub { sub: false, ra, .. } => {
                if ra == 31 {
                    "mul".into()
                } else {
                    "madd".into()
                }
            }
            Instr::MaddSub { sub: true, .. } => "msub".into(),
            Instr::Div { signed, .. } => if signed { "sdiv" } else { "udiv" }.into(),
            Instr::ShiftReg { kind, .. } => match kind {
                ShiftType::Lsl => "lsl",
                ShiftType::Lsr => "lsr",
                ShiftType::Asr => "asr",
                ShiftType::Ror => "ror",
            }
            .into(),
            Instr::LoadStore {
                load,
                size,
                signed,
                mode,
                ..
            } => {
                let base = match (load, signed, size) {
                    (true, false, 1) => "ldrb",
                    (true, false, 2) => "ldrh",
                    (true, false, _) => "ldr",
                    (true, true, 1) => "ldrsb",
                    (true, true, 2) => "ldrsh",
                    (true, true, _) => "ldrsw",
                    (false, _, 1) => "strb",
                    (false, _, 2) => "strh",
                    (false, _, _) => "str",
                };
                if matches!(mode, AddressMode::Unscaled { .. }) && load && base == "ldr" {
                    return "ldur".into();
                }
                if matches!(mode, AddressMode::Unscaled { .. }) && !load && base == "str" {
                    return "stur".into();
                }
                if matches!(mode, AddressMode::Unscaled { .. }) {
                    // LDURB / LDURSB / STURB and the like.
                    return match (load, signed, size) {
                        (true, true, 1) => "ldursb".into(),
                        (true, true, 2) => "ldursh".into(),
                        (true, true, _) => "ldursw".into(),
                        (true, false, 1) => "ldurb".into(),
                        (true, false, 2) => "ldurh".into(),
                        (false, _, 1) => "sturb".into(),
                        (false, _, 2) => "sturh".into(),
                        _ => base.into(),
                    };
                }
                base.into()
            }
            Instr::LoadStorePair { load, .. } => if load { "ldp" } else { "stp" }.into(),
            Instr::Svc { .. } => "svc".into(),
            Instr::Mrs { .. } => "mrs".into(),
            Instr::Msr { .. } => "msr".into(),
            Instr::FpArith { op, .. } => match op {
                FpOp::Add => "fadd".into(),
                FpOp::Sub => "fsub".into(),
                FpOp::Mul => "fmul".into(),
                FpOp::Div => "fdiv".into(),
                FpOp::Neg => "fneg".into(),
            },
            Instr::FpMov { .. }
            | Instr::FpMovImm { .. }
            | Instr::FpToGen { .. }
            | Instr::GenToFp { .. } => "fmov".into(),
            Instr::FpFused { op, .. } => match op {
                0 => "fmadd",
                1 => "fmsub",
                2 => "fnmadd",
                _ => "fnmsub",
            }
            .into(),
            Instr::FpLoadStore { load, size, .. } => match (load, size) {
                (true, 1) => "ldr",
                (true, 2) => "ldr",
                (true, 4) => "ldr",
                (true, 8) => "ldr",
                (true, _) => "ldr",
                (false, _) => "str",
            }
            .into(),
            Instr::Unknown { .. } => "unknown".into(),
        }
    }
}
