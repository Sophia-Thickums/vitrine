//! Decoder coverage over real compiled code.
//!
//! Random 32-bit words are mostly not instructions, so "what fraction of random words decode" is a
//! meaningless number. This reads a `.text` blob produced by a real compiler and asks two
//! questions that are worth answering:
//!
//! 1. **What fraction of instructions in real code does the decoder recognise?**
//! 2. **Which mnemonics does it miss, ranked?** — so the next class to implement is chosen from
//!    evidence rather than from what is interesting to write.
//!
//! ```sh
//! clang --target=aarch64-linux-gnu -O2 -c prog.c -o prog.o
//! llvm-objcopy -O binary --only-section=.text prog.o prog.text
//! cargo run --release --example coverage -- prog.text
//! ```
//!
//! The mnemonic for words the decoder rejects comes from `llvm-mc`, so the ranking names the
//! instruction the ISA actually has, not my guess at it.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use vitrine::a64;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: coverage <file.text>");
        std::process::exit(2);
    });
    let bytes = std::fs::read(&path).expect("read the .text blob");
    if !bytes.len().is_multiple_of(4) {
        eprintln!("warning: {} bytes is not a whole number of instructions", bytes.len());
    }
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();

    let mut decoded = 0usize;
    let mut missed: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown_words = Vec::new();

    for &word in &words {
        let instr = a64::decode(word);
        if matches!(instr, a64::Instr::Unknown { .. }) {
            unknown_words.push(word);
            let name = reference_mnemonic(word);
            *missed.entry(name).or_insert(0) += 1;
        } else {
            decoded += 1;
        }
    }

    println!(
        "{path}: {} instructions, {decoded} decoded ({:.1}%), {} unknown",
        words.len(),
        decoded as f64 / words.len() as f64 * 100.0,
        words.len() - decoded
    );

    if !missed.is_empty() {
        println!("\nmissing classes, most frequent first:");
        let mut ranked: Vec<(&String, &usize)> = missed.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (name, count) in ranked {
            println!("  {count:4}  {name}");
        }
        println!("\nfirst few unknown words:");
        for w in unknown_words.iter().take(8) {
            println!("  {w:#010x}  {}", reference_mnemonic(*w));
        }
    }
}

/// Ask `llvm-mc` what the word is, for naming purposes only.
fn reference_mnemonic(word: u32) -> String {
    let bytes = format!(
        "0x{:02x} 0x{:02x} 0x{:02x} 0x{:02x}",
        word & 0xFF,
        (word >> 8) & 0xFF,
        (word >> 16) & 0xFF,
        (word >> 24) & 0xFF
    );
    let Ok(mut child) = Command::new("llvm-mc")
        .args(["-triple=aarch64", "-disassemble"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return "?".into();
    };
    if child
        .stdin
        .as_mut()
        .map(|s| s.write_all(bytes.as_bytes()).is_err())
        .unwrap_or(true)
    {
        return "?".into();
    }
    let Ok(out) = child.wait_with_output() else {
        return "?".into();
    };
    if !out.stderr.is_empty() {
        return "(not an instruction)".into();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('.'))
        .and_then(|l| l.split_whitespace().next())
        .unwrap_or("?")
        .to_string()
}
