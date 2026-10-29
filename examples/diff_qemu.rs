//! Differential check: the reference interpreter against `qemu-aarch64` on real machine code.
//!
//! For each program below:
//!   1. assemble it with `llvm-mc`
//!   2. link it as a static aarch64 ELF and run it under `qemu-aarch64`
//!   3. load the *same instruction words* into the interpreter and run until the `SVC`
//!   4. require the same `X0`, which is the exit status
//!
//! This is the oracle that matters. A test written from my own understanding of A64 can only
//! confirm my understanding; qemu runs the real semantics. When they disagree, qemu is right.
//!
//! ```sh
//! cargo run --release --example diff_qemu
//! ```

use std::io::Write;
use std::process::{Command, Stdio};

use vitrine::interp::{Machine, Trap};

struct Case {
    name: &'static str,
    /// Assembly, ending in the exit syscall so qemu yields a status.
    source: &'static str,
}

const EPILOGUE: &str = "\n    mov x8, #93\n    svc #0\n";

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "add_immediate",
            source: "mov x0, #7\n    add x0, x0, #5",
        },
        Case {
            name: "sub_and_wrap",
            source: "mov x0, #10\n    sub x0, x0, #13",
        },
        Case {
            name: "shifted_register",
            source: "mov x1, #3\n    mov x2, #4\n    lsl x0, x1, #5\n    add x0, x0, x2",
        },
        Case {
            name: "multiply",
            source: "mov x1, #6\n    mov x2, #7\n    mul x0, x1, x2",
        },
        Case {
            name: "divide",
            source: "mov x1, #100\n    mov x2, #9\n    udiv x0, x1, x2",
        },
        Case {
            name: "divide_by_zero_is_zero",
            source: "mov x1, #100\n    mov x2, #0\n    udiv x0, x1, x2\n    add x0, x0, #1",
        },
        Case {
            name: "compare_and_branch",
            source: "mov x1, #5\n    mov x0, #0\n    cmp x1, #5\n    b.ne #8\n    add x0, x0, #9",
        },
        Case {
            name: "loop_with_cbz",
            source: "mov x1, #4\n    mov x0, #0\n    add x0, x0, #3\n    sub x1, x1, #1\n    cbnz x1, #-8",
        },
        Case {
            name: "load_store_pair",
            source: "mov x1, #0x11\n    mov x2, #0x22\n    sub sp, sp, #16\n    stp x1, x2, [sp]\n    ldp x3, x4, [sp]\n    add sp, sp, #16\n    add x0, x3, x4",
        },
        Case {
            name: "bitmask_immediate",
            source: "mov x1, #0xFF\n    and x0, x1, #0xF0",
        },
        Case {
            name: "movz_movk",
            source: "movz x0, #0x5678\n    movk x0, #0x1234, lsl #16\n    lsr x0, x0, #16",
        },
        Case {
            name: "ubfm_lsr",
            source: "mov x0, #0xF0\n    lsr x0, x0, #4",
        },
        Case {
            name: "conditional_select",
            source: "mov x1, #40\n    mov x2, #2\n    cmp x1, #40\n    csel x0, x1, x2, eq",
        },
        Case {
            name: "signed_compare",
            source: "mov x1, #-5\n    mov x0, #0\n    cmp x1, #0\n    b.ge #8\n    add x0, x0, #7",
        },
        Case {
            name: "sdiv_negative",
            source: "mov x1, #-100\n    mov x2, #7\n    sdiv x0, x1, x2\n    neg_ignore:\n    add x0, x0, #20",
        },
    ]
}

fn main() {
    let mut pass = 0;
    let mut fail = 0;

    for case in cases() {
        let asm = format!("{}{}", case.source, EPILOGUE);
        match run_case(&asm) {
            Ok((qemu, mine)) => {
                if qemu == mine {
                    println!("ok    {:<28} x0={qemu}", case.name);
                    pass += 1;
                } else {
                    println!("FAIL  {:<28} qemu={qemu} interp={mine}", case.name);
                    fail += 1;
                }
            }
            Err(e) => {
                println!("ERROR {:<28} {e}", case.name);
                fail += 1;
            }
        }
    }

    println!("\n{pass} passed, {fail} failed");
    if fail != 0 {
        std::process::exit(1);
    }
}

fn run_case(asm: &str) -> Result<(i32, i32), String> {
    // 1. encode
    let mut enc = Command::new("llvm-mc")
        .args(["-triple=aarch64", "-show-encoding"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("llvm-mc: {e}"))?;
    enc.stdin
        .as_mut()
        .ok_or("stdin")?
        .write_all(asm.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = enc.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.stderr.is_empty() {
        return Err(format!(
            "assembler rejected the program: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let mut words = Vec::new();
    for line in text.lines() {
        let Some(start) = line.find("encoding: [") else { continue };
        let end = line[start..].find(']').ok_or("bad encoding")? + start;
        let bytes: Vec<u8> = line[start + 11..end]
            .split(',')
            .filter_map(|b| u8::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        if bytes.len() != 4 {
            // This is exactly how the first run of this harness silently lost a `cbnz`: the
            // assembler printed a relocation (`[0bAAA00001,A,A,0xb5]`) for a label branch, the
            // bytes did not parse, and the program ran one iteration short. A skipped
            // instruction must be an error, not a shorter program.
            return Err(format!(
                "unresolved encoding in assembler output: {}",
                &line[start..end]
            ));
        }
        words.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    if words.is_empty() {
        return Err("assembler produced no instructions".into());
    }

    // 2. link and run under qemu
    let dir = std::env::temp_dir().join("vitrine-qemu");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let asm_path = dir.join("case.s");
    std::fs::write(
        &asm_path,
        format!(".globl _start\n_start:\n{asm}\n"),
    )
    .map_err(|e| e.to_string())?;
    let elf = dir.join("case.elf");
    let status = Command::new("clang")
        .args([
            "--target=aarch64-linux-gnu",
            "-fuse-ld=lld",
            "-nostdlib",
            "-static",
            "-Wl,-e,_start",
            "-o",
        ])
        .arg(&elf)
        .arg(&asm_path)
        .status()
        .map_err(|e| format!("clang: {e}"))?;
    if !status.success() {
        return Err("link failed".into());
    }
    let qemu = Command::new("qemu-aarch64")
        .arg(&elf)
        .status()
        .map_err(|e| format!("qemu-aarch64: {e}"))?;
    let qemu_code = qemu.code().ok_or("qemu reported no exit code")?;

    // 3. the same words in the interpreter
    let mut machine = Machine::new(0, 0x2000);
    machine.load_code(0, &words);
    // A stack, because some cases push. A64 requires 16-byte alignment; put it near the top.
    machine.sp = 0x1800;
    let trap = machine.run(100_000);
    // The syscall number travels in x8; the SVC immediate is always 0 here.
    match trap {
        Err(Trap::SupervisorCall { .. }) => {}
        other => return Err(format!("interpreter stopped early: {other:?}")),
    }
    if machine.x[8] != 93 {
        return Err(format!("expected exit syscall, x8 = {}", machine.x[8]));
    }
    // qemu reports the status as a 0..=255 byte.
    let mine = machine.x[0] as u32;
    let mine = (mine & 0xFF) as i32;
    Ok((qemu_code, mine))
}
