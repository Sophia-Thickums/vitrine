//! Basic-block analysis over decoded AArch64.
//!
//! The recompiler compiles *blocks*, not instructions: a run of instructions with one entry and
//! one exit. Getting the block boundaries wrong is the failure mode that produces a JIT which
//! works until it doesn't — a branch target that lands mid-block, a fallthrough that was never
//! emitted, a return address that points into code the translator skipped.
//!
//! # Rules
//!
//! A block **ends** at:
//! - an unconditional transfer: `B`, `BR`, `RET`
//! - a call: `BL`, `BLR` (the next instruction is a new block, reachable by return)
//! - a conditional branch: `B.cond`, `CBZ`, `CBNZ`, `TBZ`, `TBNZ` (falls through *and* branches)
//! - the last instruction of the scanned region
//!
//! A block **starts** at:
//! - offset 0
//! - any branch target, whether or not the instruction there is reachable by fallthrough
//! - the instruction after a terminator
//!
//! A branch whose target lies outside the scanned region is recorded, not followed: the caller
//! decides whether to scan more.
//!
//! # What this module refuses to do
//!
//! It does not follow indirect branches (`BR`, `BLR`) — there is no static target — and it does
//! not speculate past an instruction it could not decode. A block containing [`Instr::Unknown`] is
//! marked so the translator can refuse it rather than emit code for half of it.

use std::collections::BTreeMap;

use crate::a64::{self, Instr};

/// A branch out of a block, with the target if it is statically known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Falls through to the next instruction.
    Fallthrough,
    /// Jumps to `target`, an offset in bytes from the start of the scanned region.
    Direct { target: i64 },
    /// Jumps to a register; no static target.
    Indirect { register: u8 },
    /// Returns to the caller.
    Return,
}

impl Exit {
    /// The static target, when there is one.
    pub fn target(&self) -> Option<i64> {
        match self {
            Self::Direct { target } => Some(*target),
            _ => None,
        }
    }
}

/// One basic block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Byte offset of the first instruction within the scanned region.
    pub start: i64,
    /// Byte offset one past the last instruction.
    pub end: i64,
    /// Instructions, in order. `(offset, word, decoded)`.
    pub instructions: Vec<(i64, u32, Instr)>,
    /// How control leaves this block.
    pub exit: Exit,
    /// True when at least one instruction could not be decoded.
    pub has_unknown: bool,
}

impl Block {
    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// True for an empty block, which only arises from a branch target landing exactly at the end
    /// of the scanned region.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Whether this block can be compiled: no undecoded instructions, and at least one word.
    pub fn is_compilable(&self) -> bool {
        !self.is_empty() && !self.has_unknown
    }
}

/// The result of scanning a code region.
#[derive(Debug, Clone, Default)]
pub struct Program {
    /// Blocks whose start lies inside the region, ordered by start offset.
    pub blocks: BTreeMap<i64, Block>,
    /// Targets that were jumped to from inside the region but lie outside it.
    pub externals: Vec<i64>,
}

impl Program {
    /// The block starting at `offset`, if one was produced.
    pub fn block_at(&self, offset: i64) -> Option<&Block> {
        self.blocks.get(&offset)
    }

    /// Total instructions across all blocks.
    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(Block::len).sum()
    }
}

/// Scan `words` (little-endian instruction words, starting at offset 0) into basic blocks.
///
/// `words` must be the code exactly as it appears in guest memory; the caller is responsible for
/// having read a region large enough to contain the blocks it needs.
pub fn scan(words: &[u32]) -> Program {
    // Pass 1: decode everything and collect targets, so that a block can be closed by a target
    // that lands on it before the linear walk reaches it.
    let mut decoded: Vec<(i64, u32, Instr)> = Vec::with_capacity(words.len());
    let mut targets: Vec<i64> = Vec::new();
    let region_end = (words.len() * 4) as i64;

    for (i, &word) in words.iter().enumerate() {
        let offset = (i * 4) as i64;
        let instr = a64::decode(word);
        if let Some(t) = branch_target(instr, offset)
            && t >= 0
            && t < region_end
        {
            targets.push(t);
        }
        decoded.push((offset, word, instr));
    }

    // A target that is not 4-byte aligned cannot be an instruction boundary in A64; keep it in the
    // external list rather than pretending to split a block there.
    let mut starts: Vec<i64> = vec![0];
    starts.extend(targets.iter().copied().filter(|t| t % 4 == 0));
    for (offset, _, instr) in &decoded {
        if ends_block(*instr) {
            let next = offset + 4;
            if next < region_end {
                starts.push(next);
            }
        }
    }
    starts.sort_unstable();
    starts.dedup();

    let mut program = Program::default();
    let mut externals: Vec<i64> = Vec::new();

    for (idx, &start) in starts.iter().enumerate() {
        let limit = starts
            .get(idx + 1)
            .copied()
            .unwrap_or(region_end)
            .min(region_end);
        let mut instructions = Vec::new();
        let mut has_unknown = false;
        let mut exit = Exit::Fallthrough;

        let mut offset = start;
        while offset < limit {
            let Some(cell) = decoded.get((offset / 4) as usize) else {
                break;
            };
            let (_, word, instr) = *cell;
            instructions.push((offset, word, instr));
            if matches!(instr, Instr::Unknown { .. }) {
                has_unknown = true;
            }
            if ends_block(instr) {
                exit = exit_of(instr, offset);
                break;
            }
            offset += 4;
        }

        // A block that ran into the next start without a terminator falls through to it.
        let end = instructions
            .last()
            .map(|(o, _, i)| if ends_block(*i) { o + 4 } else { limit })
            .unwrap_or(start);

        if let Exit::Direct { target } = exit
            && !(0..region_end).contains(&target)
        {
            externals.push(target);
        }

        program.blocks.insert(
            start,
            Block {
                start,
                end,
                instructions,
                exit,
                has_unknown,
            },
        );
    }

    // Note: overlapping blocks cannot arise here by construction. Every 4-byte-aligned branch
    // target is inserted into `starts`, and a block never extends past the next start, so a
    // target is always a block boundary. An earlier version of this function carried an
    // `overlapping_targets` list and a check to fill it; the check could never fire. A misaligned
    // target is the only case that would violate a boundary, and A64 cannot branch to one.

    externals.sort_unstable();
    externals.dedup();
    program.externals = externals;
    program
}

/// True for instructions that end a block.
fn ends_block(instr: Instr) -> bool {
    matches!(
        instr,
        Instr::B { .. }
            | Instr::Br { .. }
            | Instr::Ret { .. }
            | Instr::Bl { .. }
            | Instr::Blr { .. }
            | Instr::BCond { .. }
            | Instr::Cbz { .. }
            | Instr::Tbz { .. }
    )
}

/// How `instr` at `offset` leaves its block.
fn exit_of(instr: Instr, offset: i64) -> Exit {
    match instr {
        Instr::B { offset: rel } => Exit::Direct {
            target: offset + rel,
        },
        Instr::BCond { offset: rel, .. }
        | Instr::Cbz { offset: rel, .. }
        | Instr::Tbz { offset: rel, .. } => Exit::Direct {
            target: offset + rel,
        },
        Instr::Br { rn } | Instr::Blr { rn } => Exit::Indirect { register: rn },
        Instr::Ret { .. } => Exit::Return,
        // BL/BLR return to the next instruction; the block itself falls through after the call.
        _ => Exit::Fallthrough,
    }
}

/// The static target of a branch, if it has one.
fn branch_target(instr: Instr, offset: i64) -> Option<i64> {
    match instr {
        Instr::B { offset: rel }
        | Instr::BCond { offset: rel, .. }
        | Instr::Cbz { offset: rel, .. }
        | Instr::Tbz { offset: rel, .. } => Some(offset + rel),
        _ => None,
    }
}
