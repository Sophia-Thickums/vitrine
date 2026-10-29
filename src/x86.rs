//! A small x86-64 encoder — just enough to translate AArch64 blocks.
//!
//! Hand-writing instruction encodings is the same trap as hand-writing AArch64 encodings: a wrong
//! ModRM byte produces an instruction that is *valid* and does something else. So this module is
//! built the same way as the decoder, in reverse: every emitted form is disassembled with
//! `llvm-mc -triple=x86_64` and compared in `tests/x86.rs`. The test is the specification.
//!
//! # Register convention for generated code
//!
//! Generated functions take one argument, a `*mut` to the machine state, in `rdi`. That pointer
//! stays in `r15` for the whole function; `rax`, `rcx`, `rdx`, `rsi` and `r8`..`r11` are scratch.
//! Guest registers live in memory (offset 0 is `x0`), so a block never has to spill: it loads what
//! it needs, computes, and stores back. That is slow and correct, which is the right order — the
//! interpreter is the oracle, and a faster register allocator can come later without changing any
//! generated semantics.
//!
//! # What is deliberately absent
//!
//! No SIB bytes, no displacement-free addressing of `rsp`/`r12`, no REX for byte registers. The
//! encoder only emits forms this project uses, and a form that cannot be expressed is a compile
//! error in the emitter, not a silently mis-encoded instruction.

/// The general-purpose registers this encoder will name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Reg {
    fn low(self) -> u8 {
        (self as u8) & 7
    }

    fn high(self) -> bool {
        (self as u8) >= 8
    }
}

/// Condition codes, in the encoding x86 uses for `jcc` and `setcc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc {
    O = 0,
    No = 1,
    B = 2,
    Ae = 3,
    E = 4,
    Ne = 5,
    Be = 6,
    A = 7,
    S = 8,
    Ns = 9,
    P = 10,
    Np = 11,
    L = 12,
    Ge = 13,
    Le = 14,
    G = 15,
}

/// A byte sink for emitted code.
#[derive(Default)]
pub struct Code {
    bytes: Vec<u8>,
    /// Offsets of `jmp`/`jcc` with a 32-bit displacement still to be filled in, and the label they
    /// want. Resolved by [`Code::finish`].
    fixups: Vec<(usize, usize)>,
    /// Offset of each label, in order.
    labels: Vec<usize>,
}

impl Code {
    /// New, empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes emitted so far.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True when nothing has been emitted.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The emitted bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The emitted code as owned bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn byte(&mut self, b: u8) {
        self.bytes.push(b);
    }

    fn push_bytes(&mut self, bs: &[u8]) {
        self.bytes.extend_from_slice(bs);
    }

    fn push_i32(&mut self, v: i32) {
        self.push_bytes(&v.to_le_bytes());
    }

    /// Declare that the next emitted byte is the target of `label`.
    pub fn label(&mut self, label: usize) {
        while self.labels.len() <= label {
            self.labels.push(usize::MAX);
        }
        self.labels[label] = self.bytes.len();
    }

    /// Resolve every branch and return the finished code.
    pub fn finish(mut self) -> Vec<u8> {
        for (offset, label) in std::mem::take(&mut self.fixups) {
            let Some(&target) = self.labels.get(label) else {
                panic!("label {label} was never placed");
            };
            if target == usize::MAX {
                panic!("label {label} was never placed");
            }
            let rel = target as i64 - (offset as i64 + 4);
            let rel = i32::try_from(rel).expect("branch displacement out of range");
            self.bytes[offset..offset + 4].copy_from_slice(&rel.to_le_bytes());
        }
        self.bytes
    }

    // -- prefixes and REX ---------------------------------------------------

    fn rex(&mut self, w: bool, r: Option<Reg>, x: bool, b: Option<Reg>) {
        let mut rex = 0x40;
        if w {
            rex |= 0x08;
        }
        if r.is_some_and(Reg::high) {
            rex |= 0x04;
        }
        if x {
            rex |= 0x02;
        }
        if b.is_some_and(Reg::high) {
            rex |= 0x01;
        }
        if rex != 0x40 {
            self.byte(rex);
        }
    }

    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.byte((mode << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    /// Emit a ModRM byte for `[base + disp]` and the smallest displacement that fits.
    ///
    /// rbp/r13 cannot use mod=00 (that form means "disp32 follows"), so a zero displacement on
    /// them still needs a byte. Choosing the short forms is not cosmetic: it is what lets the
    /// encoder be compared byte-for-byte against the assembler, because the assembler picks them
    /// too.
    fn modrm_disp(&mut self, reg: u8, base: Reg, disp: i32) {
        let needs_disp32 = disp == 0 && matches!(base, Reg::Rbp | Reg::R13);
        if disp == 0 && !needs_disp32 {
            self.modrm(0b00, reg, base.low());
        } else if let Ok(d8) = i8::try_from(disp) {
            self.modrm(0b01, reg, base.low());
            self.byte(d8 as u8);
        } else {
            self.modrm(0b10, reg, base.low());
            self.push_i32(disp);
        }
    }

    // -- moves --------------------------------------------------------------

    /// `mov r64, imm64`.
    pub fn mov_imm64(&mut self, dst: Reg, value: u64) {
        self.rex(true, None, false, Some(dst));
        self.byte(0xB8 + dst.low());
        self.push_bytes(&value.to_le_bytes());
    }

    /// `mov r64, [base + disp32]`.
    pub fn mov_load(&mut self, dst: Reg, base: Reg, disp: i32) {
        assert!(
            base.low() != 4,
            "rsp/r12 as a base needs a SIB byte, which this encoder does not emit"
        );
        self.rex(true, Some(dst), false, Some(base));
        self.byte(0x8B);
        self.modrm_disp(dst.low(), base, disp);
    }

    /// `mov [base + disp32], src`.
    pub fn mov_store(&mut self, base: Reg, disp: i32, src: Reg) {
        assert!(
            base.low() != 4,
            "rsp/r12 as a base needs a SIB byte, which this encoder does not emit"
        );
        self.rex(true, Some(src), false, Some(base));
        self.byte(0x89);
        self.modrm_disp(src.low(), base, disp);
    }

    /// `mov r64, r64`.
    pub fn mov_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x89);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `movzx r32, byte [base + disp32]` — used for 8-bit guest loads.
    pub fn movzx_u8_load(&mut self, dst: Reg, base: Reg, disp: i32) {
        assert!(base.low() != 4, "SIB base not supported");
        self.rex(false, Some(dst), false, Some(base));
        self.push_bytes(&[0x0F, 0xB6]);
        self.modrm_disp(dst.low(), base, disp);
    }

    /// `movsx r64, byte [base + disp32]` — `LDRSB`.
    pub fn movsx_i8_load(&mut self, dst: Reg, base: Reg, disp: i32) {
        assert!(base.low() != 4, "SIB base not supported");
        self.rex(true, Some(dst), false, Some(base));
        self.push_bytes(&[0x0F, 0xBE]);
        self.modrm_disp(dst.low(), base, disp);
    }

    /// `movsxd r64, r32` — widen a 32-bit guest value.
    pub fn movsxd(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(dst), false, Some(src));
        self.byte(0x63);
        self.modrm(0b11, dst.low(), src.low());
    }

    // -- arithmetic ---------------------------------------------------------

    /// `add r64, r64`.
    pub fn add_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x01);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `add r64, imm32` (sign-extended).
    pub fn add_imm(&mut self, dst: Reg, imm: i32) {
        if let Ok(b) = i8::try_from(imm) {
            self.rex(true, None, false, Some(dst));
            self.byte(0x83);
            self.modrm(0b11, 0, dst.low());
            self.byte(b as u8);
            return;
        }
        self.rex(true, None, false, Some(dst));
        self.byte(0x81);
        self.modrm(0b11, 0, dst.low());
        self.push_i32(imm);
    }

    /// `sub r64, r64`.
    pub fn sub_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x29);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `sub r64, imm32`.
    pub fn sub_imm(&mut self, dst: Reg, imm: i32) {
        if let Ok(b) = i8::try_from(imm) {
            self.rex(true, None, false, Some(dst));
            self.byte(0x83);
            self.modrm(0b11, 5, dst.low());
            self.byte(b as u8);
            return;
        }
        self.rex(true, None, false, Some(dst));
        self.byte(0x81);
        self.modrm(0b11, 5, dst.low());
        self.push_i32(imm);
    }

    /// `cmp r64, r64`.
    pub fn cmp_reg(&mut self, a: Reg, b: Reg) {
        self.rex(true, Some(b), false, Some(a));
        self.byte(0x39);
        self.modrm(0b11, b.low(), a.low());
    }

    /// `cmp r64, imm32`.
    pub fn cmp_imm(&mut self, dst: Reg, imm: i32) {
        if let Ok(b) = i8::try_from(imm) {
            self.rex(true, None, false, Some(dst));
            self.byte(0x83);
            self.modrm(0b11, 7, dst.low());
            self.byte(b as u8);
            return;
        }
        self.rex(true, None, false, Some(dst));
        self.byte(0x81);
        self.modrm(0b11, 7, dst.low());
        self.push_i32(imm);
    }

    /// `test r64, r64`.
    pub fn test_reg(&mut self, a: Reg, b: Reg) {
        self.rex(true, Some(b), false, Some(a));
        self.byte(0x85);
        self.modrm(0b11, b.low(), a.low());
    }

    /// `and r64, r64`.
    pub fn and_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x21);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `or r64, r64`.
    pub fn or_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x09);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `xor r64, r64`.
    pub fn xor_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(src), false, Some(dst));
        self.byte(0x31);
        self.modrm(0b11, src.low(), dst.low());
    }

    /// `neg r64`.
    pub fn neg(&mut self, r: Reg) {
        self.rex(true, None, false, Some(r));
        self.byte(0xF7);
        self.modrm(0b11, 3, r.low());
    }

    /// `not r64`.
    pub fn not(&mut self, r: Reg) {
        self.rex(true, None, false, Some(r));
        self.byte(0xF7);
        self.modrm(0b11, 2, r.low());
    }

    /// `imul r64, r64`.
    pub fn imul_reg(&mut self, dst: Reg, src: Reg) {
        self.rex(true, Some(dst), false, Some(src));
        self.push_bytes(&[0x0F, 0xAF]);
        self.modrm(0b11, dst.low(), src.low());
    }

    /// `cqo` — sign-extend `rax` into `rdx:rax`.
    pub fn cqo(&mut self) {
        self.push_bytes(&[0x48, 0x99]);
    }

    /// `idiv r64` (signed) or `div r64` (unsigned).
    pub fn div(&mut self, r: Reg, signed: bool) {
        self.rex(true, None, false, Some(r));
        self.byte(0xF7);
        self.modrm(0b11, if signed { 7 } else { 6 }, r.low());
    }

    /// `cqo` counterpart for unsigned division: zero `rdx`.
    pub fn zero_rdx(&mut self) {
        self.xor_reg(Reg::Rdx, Reg::Rdx);
    }

    /// `shl`/`shr`/`sar r64, imm8`.
    pub fn shift_imm(&mut self, r: Reg, kind: u8, amount: u8) {
        self.rex(true, None, false, Some(r));
        self.byte(0xC1);
        self.modrm(0b11, kind, r.low());
        self.byte(amount);
    }

    /// `shl`/`shr`/`sar r64, cl`.
    pub fn shift_cl(&mut self, r: Reg, kind: u8) {
        self.rex(true, None, false, Some(r));
        self.byte(0xD3);
        self.modrm(0b11, kind, r.low());
    }

    /// `ror r64, imm8`.
    pub fn ror_imm(&mut self, r: Reg, amount: u8) {
        self.rex(true, None, false, Some(r));
        self.byte(0xC1);
        self.modrm(0b11, 1, r.low());
        self.byte(amount);
    }

    // -- scalar double-precision (SSE2) ------------------------------------

    /// `movsd xmm, [base + disp]`.
    pub fn movsd_load(&mut self, xmm: u8, base: Reg, disp: i32) {
        assert!(base.low() != 4, "SIB base not supported");
        self.byte(0xF2); // mandatory prefix, BEFORE any REX
        self.sse_rex(xmm, base);
        self.byte(0x0F);
        self.byte(0x10);
        self.modrm_disp(xmm & 7, base, disp);
    }

    /// `movsd [base + disp], xmm`.
    pub fn movsd_store(&mut self, base: Reg, disp: i32, xmm: u8) {
        assert!(base.low() != 4, "SIB base not supported");
        self.byte(0xF2);
        self.sse_rex(xmm, base);
        self.push_bytes(&[0x0F, 0x11]);
        self.modrm_disp(xmm & 7, base, disp);
    }

    /// `movsd xmm, xmm`.
    pub fn movsd_reg(&mut self, dst: u8, src: u8) {
        self.byte(0xF2);
        self.sse_rex2(dst, src);
        self.push_bytes(&[0x0F, 0x10]);
        self.modrm(0b11, dst & 7, src & 7);
    }

    /// `addsd`/`subsd`/`mulsd`/`divsd xmm, xmm`.
    pub fn sd_arith(&mut self, opcode: u8, dst: u8, src: u8) {
        self.byte(0xF2);
        self.sse_rex2(dst, src);
        self.byte(0x0F);
        self.byte(opcode);
        self.modrm(0b11, dst & 7, src & 7);
    }

    /// `movq xmm, r64`.
    ///
    /// The 0x66 operand-size prefix is a *legacy* prefix and must come BEFORE the REX byte, and REX
    /// does carry W here because the operand is 64-bit. Emitted the other way round it is a valid
    /// instruction with a different meaning.
    pub fn movq_to_xmm(&mut self, xmm: u8, gp: Reg) {
        self.byte(0x66);
        self.sse_rex_w(xmm, gp);
        self.push_bytes(&[0x0F, 0x6E]);
        self.modrm(0b11, xmm & 7, gp.low());
    }

    /// `movq r64, xmm`.
    pub fn movq_from_xmm(&mut self, gp: Reg, xmm: u8) {
        self.byte(0x66);
        self.sse_rex_w(xmm, gp);
        self.push_bytes(&[0x0F, 0x7E]);
        self.modrm(0b11, xmm & 7, gp.low());
    }

    /// REX for a scalar-SSE instruction: no W bit, R for the XMM in `reg`, B for a general base.
    fn sse_rex(&mut self, xmm: u8, base: Reg) {
        let mut rex = 0x40;
        if xmm >= 8 {
            rex |= 0x04;
        }
        if base.high() {
            rex |= 0x01;
        }
        if rex != 0x40 {
            self.byte(rex);
        }
    }

    /// REX for `xmm, xmm`: R for the destination, B for the source.
    fn sse_rex2(&mut self, dst: u8, src: u8) {
        let mut rex = 0x40;
        if dst >= 8 {
            rex |= 0x04;
        }
        if src >= 8 {
            rex |= 0x01;
        }
        if rex != 0x40 {
            self.byte(rex);
        }
    }

    /// REX.W for the 64-bit `movq` forms.
    fn sse_rex_w(&mut self, xmm: u8, gp: Reg) {
        let mut rex = 0x48;
        if xmm >= 8 {
            rex |= 0x04;
        }
        if gp.high() {
            rex |= 0x01;
        }
        self.byte(rex);
    }

    // -- flags and conditionals --------------------------------------------

    /// `setcc r8`.
    ///
    /// Byte registers 4..=7 (spl/bpl/sil/dil) need a REX prefix to be addressable at all, and this
    /// encoder refuses them rather than emitting an instruction that silently means ah/ch/dh/bh.
    /// Registers 8..15 need REX.B, which is emitted here.
    pub fn setcc(&mut self, cc: Cc, dst: Reg) {
        assert!(
            !matches!(dst, Reg::Rsp | Reg::Rbp | Reg::Rsi | Reg::Rdi),
            "byte form of rsp/rbp/rsi/rdi needs REX, which this encoder does not emit"
        );
        if dst.high() {
            self.byte(0x41);
        }
        self.push_bytes(&[0x0F, 0x90 + cc as u8]);
        self.modrm(0b11, 0, dst.low());
    }

    /// `movzx r32, r8`.
    ///
    /// A REX prefix is required when either side is r8..r15 — without it `movzx ecx, r8b` encodes
    /// as `movzx ecx, al`, a valid instruction that reads the wrong register. That exact omission
    /// made the flag packer read `al` instead of the captured `r8b` and produced a wrong carry.
    pub fn movzx_u8(&mut self, dst: Reg, src: Reg) {
        if dst.high() || src.high() {
            let mut rex = 0x40;
            if dst.high() {
                rex |= 0x04;
            }
            if src.high() {
                rex |= 0x01;
            }
            self.byte(rex);
        }
        self.push_bytes(&[0x0F, 0xB6]);
        self.modrm(0b11, dst.low(), src.low());
    }

    /// `jmp rel32` to `label`.
    pub fn jmp(&mut self, label: usize) {
        self.byte(0xE9);
        self.fixups.push((self.bytes.len(), label));
        self.push_i32(0);
    }

    /// `jcc rel32` to `label`.
    pub fn jcc(&mut self, cc: Cc, label: usize) {
        self.push_bytes(&[0x0F, 0x80 + cc as u8]);
        self.fixups.push((self.bytes.len(), label));
        self.push_i32(0);
    }

    /// `jmp r64`.
    pub fn jmp_reg(&mut self, r: Reg) {
        self.rex(false, None, false, Some(r));
        self.byte(0xFF);
        self.modrm(0b11, 4, r.low());
    }

    /// `ret`.
    pub fn ret(&mut self) {
        self.byte(0xC3);
    }

    /// `push r64`.
    pub fn push_reg(&mut self, r: Reg) {
        if r.high() {
            self.byte(0x41);
        }
        self.byte(0x50 + r.low());
    }

    /// `pop r64`.
    pub fn pop(&mut self, r: Reg) {
        if r.high() {
            self.byte(0x41);
        }
        self.byte(0x58 + r.low());
    }

    /// `ud2` — the deliberate trap for a path that must not be reached.
    pub fn ud2(&mut self) {
        self.push_bytes(&[0x0F, 0x0B]);
    }
}

/// Offsets into the machine-state block that generated code addresses directly.
///
/// These mirror [`crate::interp::Machine`]'s field order deliberately: the JIT and the interpreter
/// must see the same layout, and a mismatch here is a silent corruption of a guest register. The
/// test `state_offsets_match_the_interpreter` checks them against `size_of` and `offset_of`.
pub mod state {
    /// `x0` .. `x30`, each 8 bytes.
    pub const X0: i32 = 0;
    /// Byte offset of guest `xN`.
    pub const fn x(n: u8) -> i32 {
        X0 + (n as i32) * 8
    }
    /// `sp`.
    pub const SP: i32 = 31 * 8;
    /// `pc`.
    pub const PC: i32 = SP + 8;
    /// Flags, packed as a byte: bit 3 = N, 2 = Z, 1 = C, 0 = V. Seven bytes of padding follow, then
    /// the guest memory base pointer.
    pub const FLAGS: i32 = PC + 8;
    /// Host pointer to guest memory byte 0 (the interpreter's `base` address maps here).
    pub const MEM_PTR: i32 = FLAGS + 8;
    /// Guest memory length in bytes.
    pub const MEM_LEN: i32 = MEM_PTR + 8;
    /// `V0..V31`, 16 bytes each.
    pub const FP: i32 = MEM_LEN + 8;
    /// Byte offset of floating-point register `n`.
    pub const fn fp(n: u8) -> i32 {
        FP + (n as i32) * 16
    }
}
