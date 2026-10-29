//! Decode a shard of words and print a checksum, for running the same work on more than one box.
//!
//! The oracle (`examples/diff_llvm`) needs `llvm-mc`; this does not. It answers two questions that
//! a single machine cannot:
//!
//! 1. **Determinism.** Two boxes decoding the same words must produce the same checksum. If they
//!    differ, something in the decoder is reading uninitialised memory, relying on unspecified
//!    layout, or has architecture-dependent behaviour.
//! 2. **Throughput.** Words per second, measured on both, which is the number a JIT's decode rate
//!    has to live inside.
//!
//! ```sh
//! cargo run --release --example shard -- <seed> <count>
//! ```

use std::time::Instant;

use vitrine::a64;

fn main() {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args
        .next()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(1);
    let count: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let threads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let start = Instant::now();

    // Split the range across threads; each takes a seed derived from the base so the work is
    // reproducible from (seed, count, threads) alone.
    let per = count.div_ceil(threads);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                let n = per.min(count.saturating_sub(t * per));
                run_shard(seed.wrapping_add(t as u64 * 0x9E37_79B9_7F4A_7C15), n)
            })
        })
        .collect();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut unknown = 0usize;
    for h in handles {
        let (h_hash, h_unknown) = h.join().expect("shard thread");
        hash ^= h_hash;
        unknown += h_unknown;
    }
    let count = count.min(per * threads).max(count);

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "seed={seed:#x} count={count} threads={threads} hash={hash:#018x} unknown={unknown} \
         elapsed={elapsed:.3}s rate={:.1}Mw/s total",
        count as f64 / elapsed / 1e6
    );
}

fn run_shard(seed: u64, count: usize) -> (u64, usize) {
    let mut state = seed | 1;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    let mut unknown = 0usize;

    for _ in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let word = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;

        let instr = a64::decode(word);
        if matches!(instr, a64::Instr::Unknown { .. }) {
            unknown += 1;
        }
        // Hash the decoded form, not the word, so a decoder change is visible in the checksum.
        let m = instr.mnemonic();
        for b in m.as_bytes() {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
        // Hash the offset field too: two decoders can agree on the mnemonic and differ here.
        let extra: i64 = match instr {
            a64::Instr::B { offset } | a64::Instr::Bl { offset } => offset,
            a64::Instr::BCond { offset, .. } => offset,
            a64::Instr::Cbz { offset, .. } | a64::Instr::Tbz { offset, .. } => offset,
            a64::Instr::Adr { offset, .. } => offset,
            a64::Instr::Adrp { page_offset, .. } => page_offset,
            _ => 0,
        };
        for b in extra.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    (hash, unknown)
}
