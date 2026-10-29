//! Differential check: this decoder against `llvm-mc`'s disassembler, over random words.
//!
//! Not a unit test, because it needs an external tool. Run it by hand:
//!
//! ```sh
//! cargo run --release --example diff_llvm -- 100000 0x1234
//! ```
//!
//! Three outcomes matter, and they are not the same thing:
//!
//! - **disagree** — both decoded, different mnemonics. A bug, always.
//! - **over-reach** — this decoder produced an instruction where `llvm-mc` says the word is not a
//!   valid instruction. Worse than a bug: it means a garbage word is compiled as if it were code.
//! - **under-reach** — `llvm-mc` decoded it and this crate said `Unknown`. Expected for the classes
//!   not implemented yet (vectors, atomics, pointer auth); a count to watch, not a failure.
//!
//! The point of running it is the first two: both must be zero.

use std::io::Write;
use std::process::{Command, Stdio};

use vitrine::a64;

fn main() {
    let mut args = std::env::args().skip(1);
    let count: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let seed: u64 = args
        .next()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);

    // xorshift64*, so the corpus is reproducible from the seed alone.
    let mut state = seed | 1;
    let mut words = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        words.push(v as u32);
    }

    let input: String = words
        .iter()
        .map(|w| {
            format!(
                "0x{:02x} 0x{:02x} 0x{:02x} 0x{:02x}",
                w & 0xFF,
                (w >> 8) & 0xFF,
                (w >> 16) & 0xFF,
                (w >> 24) & 0xFF
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut child = Command::new("llvm-mc")
        .args(["-triple=aarch64", "-disassemble"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("llvm-mc must be on PATH for this check");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write to llvm-mc");
    let out = child.wait_with_output().expect("llvm-mc");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    // llvm-mc prints one line per decoded word and an error block per rejected word, so the
    // decoded list is shorter than the input. Recover per-word validity by counting errors.
    let mnemonics: Vec<&str> = stdout
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('.') && !t.starts_with("//")
        })
        .filter_map(|l| l.split_whitespace().next())
        .filter(|m| !m.starts_with("0x"))
        .collect();

    let rejected: usize = stderr.matches("invalid instruction encoding").count()
        + stderr.matches("potentially undefined instruction encoding").count();

    println!("words={count} llvm_decoded={} llvm_rejected={rejected}", mnemonics.len());

    // Re-run word by word whenever the bulk counts do not line up, so the comparison is exact
    // rather than positional guesswork.
    let mut disagree = 0usize;
    let mut over_reach = 0usize;
    let mut under_reach = 0usize;
    let mut agree = 0usize;
    let mut examples: Vec<String> = Vec::new();

    for &w in &words {
        let mine = a64::decode(w);
        let mine_m = mine.mnemonic();
        let reference = disassemble_one(w);
        match reference {
            None => {
                // llvm says: not an instruction.
                if !matches!(mine, a64::Instr::Unknown { .. }) {
                    over_reach += 1;
                    if examples.len() < 20 {
                        examples.push(format!("{w:#010x}: llvm rejects, decoded as `{mine_m}`"));
                    }
                } else {
                    agree += 1;
                }
            }
            Some(m) if m == "invalid" => {
                if !matches!(mine, a64::Instr::Unknown { .. }) {
                    over_reach += 1;
                    if examples.len() < 20 {
                        examples.push(format!("{w:#010x}: llvm rejects, decoded as `{mine_m}`"));
                    }
                } else {
                    agree += 1;
                }
            }
            Some(ref_m) => {
                if matches!(mine, a64::Instr::Unknown { .. }) {
                    under_reach += 1;
                } else if same_class(&mine_m, &ref_m) {
                    agree += 1;
                } else {
                    disagree += 1;
                    if examples.len() < 20 {
                        examples.push(format!("{w:#010x}: llvm `{ref_m}`, mine `{mine_m}`"));
                    }
                }
            }
        }
    }

    println!("agree={agree} disagree={disagree} over_reach={over_reach} under_reach={under_reach}");
    for e in &examples {
        println!("  {e}");
    }
    if disagree != 0 || over_reach != 0 {
        std::process::exit(1);
    }
}

/// Compare mnemonics, tolerating the aliasing differences that are not errors: llvm prints `mov`
/// where the encoding is `movz`, and size-suffixed forms share a stem.
fn same_class(mine: &str, reference: &str) -> bool {
    if mine == reference {
        return true;
    }
    let stem = |m: &str| m.trim_end_matches(['b', 'h', 's', 'w']).to_string();
    if stem(mine) == stem(reference) {
        return true;
    }
    matches!(
        (mine, reference),
        ("mov", "movz") | ("mov", "movn") | ("movz", "mov") | ("movn", "mov")
    )
}

fn disassemble_one(word: u32) -> Option<String> {
    let bytes = format!(
        "0x{:02x} 0x{:02x} 0x{:02x} 0x{:02x}",
        word & 0xFF,
        (word >> 8) & 0xFF,
        (word >> 16) & 0xFF,
        (word >> 24) & 0xFF
    );
    let mut child = Command::new("llvm-mc")
        .args(["-triple=aarch64", "-disassemble"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(bytes.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.stderr.is_empty() {
        return Some("invalid".to_string());
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('.'))
        .and_then(|l| l.split_whitespace().next())
        .map(str::to_string)
}
