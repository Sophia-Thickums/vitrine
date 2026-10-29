//! Interpreted versus compiled: the number that justifies a JIT at all.
//!
//! Same program, same state, two engines. The JIT here compiles each block on every entry — there
//! is no block cache yet — so the figure below includes the whole translation cost and is a floor,
//! not a ceiling. When it still beats the interpreter, the semantics are cheap enough that the
//! remaining work is bookkeeping.
//!
//! ```sh
//! cargo run --release --example bench
//! ```

use std::collections::BTreeMap;
use std::time::Instant;

use vitrine::interp::Machine;
use vitrine::jit::{self, JitState};

fn programs() -> BTreeMap<String, Vec<u32>> {
    let raw = include_str!("../tests/fixtures/programs.tsv");
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

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "loop_1000".to_string());
    let repeats: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let words = programs()
        .remove(&name)
        .unwrap_or_else(|| panic!("no fixture named {name}"));

    // Interpreted
    let start = Instant::now();
    let mut retired = 0u64;
    for _ in 0..repeats {
        let mut m = Machine::new(0, 0x2000);
        m.load_code(0, &words);
        m.sp = 0x1800;
        let mut steps = 0u64;
        while m.step().is_ok() {
            steps += 1;
            if steps > 20_000 {
                break;
            }
        }
        retired += steps;
    }
    let interp_elapsed = start.elapsed().as_secs_f64();

    // Compiled, with a warm block cache: the first run pays translation, the rest reuse it.
    let start = Instant::now();
    let mut blocks = 0u64;
    let mut cache = jit::BlockCache::new();
    for _ in 0..repeats {
        let mut memory = vec![0u8; 0x2000];
        let mut state = JitState::new(0, 0);
        state.sp = 0x1800;
        blocks += jit::run_cached(&words, &mut memory, &mut state, 20_000, &mut cache)
            .unwrap_or_else(|e| panic!("jit: {e}"));
    }
    let jit_elapsed = start.elapsed().as_secs_f64();
    let cached = cache.len();

    println!(
        "program: {name}  ({} instructions, {repeats} runs)",
        words.len()
    );
    println!(
        "interpreter: {interp_elapsed:.3}s  {retired} instructions retired  {:.1} M/s",
        retired as f64 / interp_elapsed / 1e6
    );
    println!(
        "compiled:    {jit_elapsed:.3}s  {blocks} block entries over {cached} cached block(s)  {:.3} us each",
        jit_elapsed / blocks as f64 * 1e6
    );
    let ratio = interp_elapsed / jit_elapsed;
    println!("\nwall-clock ratio (interpreter / compiled) = {ratio:.2}x");
    println!(
        "Blocks are compiled once and reused; the translation cost is amortised over the run."
    );
}
