//! The x86-64 encoder against the disassembler.
//!
//! Every form the emitter can produce is emitted here, disassembled with `llvm-mc -triple=x86_64`
//! and compared to what it was meant to be. An encoder is a place where a wrong ModRM byte yields
//! a valid instruction that does something else, so the check has to be external: this file states
//! the intent, the assembler states the truth.
//!
//! ```sh
//! cargo run --release --example diff_x86
//! ```

use std::io::Write;
use std::process::{Command, Stdio};

use vitrine::x86::{Cc, Code, Reg};

fn main() {
    let mut failures = 0;

    // (description, emitted bytes, the assembly the bytes must be)
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("mov rax, imm64", emit(|c| c.mov_imm64(Reg::Rax, 0x1122_3344_5566_7788)), "movabsq $0x1122334455667788, %rax"),
        ("mov r15, imm64", emit(|c| c.mov_imm64(Reg::R15, 1)), "movabsq $1, %r15"),
        ("mov rax, [rdi]", emit(|c| c.mov_load(Reg::Rax, Reg::Rdi, 0)), "mov (%rdi), %rax"),
        ("mov rax, [rdi+8]", emit(|c| c.mov_load(Reg::Rax, Reg::Rdi, 8)), "mov 8(%rdi), %rax"),
        ("mov [r15-16], rax", emit(|c| c.mov_store(Reg::R15, -16, Reg::Rax)), "mov %rax, -16(%r15)"),
        ("mov rcx, rdx", emit(|c| c.mov_reg(Reg::Rcx, Reg::Rdx)), "mov %rdx, %rcx"),
        ("add rax, rcx", emit(|c| c.add_reg(Reg::Rax, Reg::Rcx)), "add %rcx, %rax"),
        ("add rax, 16", emit(|c| c.add_imm(Reg::Rax, 16)), "add $16, %rax"),
        ("sub rsp, 16", emit(|c| c.sub_imm(Reg::Rsp, 16)), "sub $16, %rsp"),
        ("sub rax, rbx", emit(|c| c.sub_reg(Reg::Rax, Reg::Rbx)), "sub %rbx, %rax"),
        ("cmp rax, rcx", emit(|c| c.cmp_reg(Reg::Rax, Reg::Rcx)), "cmp %rcx, %rax"),
        ("cmp rax, 0", emit(|c| c.cmp_imm(Reg::Rax, 0)), "cmp $0, %rax"),
        ("test rax, rax", emit(|c| c.test_reg(Reg::Rax, Reg::Rax)), "test %rax, %rax"),
        ("and rax, rcx", emit(|c| c.and_reg(Reg::Rax, Reg::Rcx)), "and %rcx, %rax"),
        ("or rax, rcx", emit(|c| c.or_reg(Reg::Rax, Reg::Rcx)), "or %rcx, %rax"),
        ("xor rdx, rdx", emit(|c| c.xor_reg(Reg::Rdx, Reg::Rdx)), "xor %rdx, %rdx"),
        ("neg rax", emit(|c| c.neg(Reg::Rax)), "neg %rax"),
        ("not rax", emit(|c| c.not(Reg::Rax)), "not %rax"),
        ("imul rax, rcx", emit(|c| c.imul_reg(Reg::Rax, Reg::Rcx)), "imul %rcx, %rax"),
        ("cqo", emit(|c| c.cqo()), "cqto"),
        ("idiv rcx", emit(|c| c.div(Reg::Rcx, true)), "idivq %rcx"),
        ("div rcx", emit(|c| c.div(Reg::Rcx, false)), "divq %rcx"),
        ("shl rax, 4", emit(|c| c.shift_imm(Reg::Rax, 4, 4)), "shl $4, %rax"),
        ("shr rax, 4", emit(|c| c.shift_imm(Reg::Rax, 5, 4)), "shr $4, %rax"),
        ("sar rax, 4", emit(|c| c.shift_imm(Reg::Rax, 7, 4)), "sar $4, %rax"),
        ("shl rax, cl", emit(|c| c.shift_cl(Reg::Rax, 4)), "shl %cl, %rax"),
        ("ror rax, 4", emit(|c| c.ror_imm(Reg::Rax, 4)), "ror $4, %rax"),
        ("sete al", emit(|c| c.setcc(Cc::E, Reg::Rax)), "sete %al"),
        ("setne cl", emit(|c| c.setcc(Cc::Ne, Reg::Rcx)), "setne %cl"),
        ("movzx eax, al", emit(|c| c.movzx_u8(Reg::Rax, Reg::Rax)), "movzbl %al, %eax"),
        ("movzx eax, byte [rdi+3]", emit(|c| c.movzx_u8_load(Reg::Rax, Reg::Rdi, 3)), "movzbl 3(%rdi), %eax"),
        ("movsx rax, byte [rdi+3]", emit(|c| c.movsx_i8_load(Reg::Rax, Reg::Rdi, 3)), "movsbq 3(%rdi), %rax"),
        ("movsxd rax, eax", emit(|c| c.movsxd(Reg::Rax, Reg::Rax)), "movslq %eax, %rax"),
        ("push rbx", emit(|c| c.push_reg(Reg::Rbx)), "pushq %rbx"),
        ("pop rbx", emit(|c| c.pop(Reg::Rbx)), "popq %rbx"),
        ("ret", emit(|c| c.ret()), "retq"),
        ("ud2", emit(|c| c.ud2()), "ud2"),
    ];

    for (name, bytes, expected) in cases {
        // The oracle is byte-exact: assemble the intended instruction and require the same bytes.
        // Comparing disassembly text is weaker — the assembler's spacing and suffix choices then
        // have to be reproduced by hand in 35 strings, which is the hand-typing trap again.
        let reference = assemble(expected);
        if reference == bytes {
            println!("ok    {name:<26} {:<28} {:02x?}", normalise(expected), bytes);
        } else {
            failures += 1;
            println!(
                "FAIL  {name:<26}\n      intended `{}`\n      assembler {:02x?}\n      emitted   {:02x?}",
                normalise(expected),
                reference,
                bytes
            );
        }
    }

    // Branches are checked by shape: a forward jmp over a ret must land after it.
    let mut c = Code::new();
    c.jmp(0);
    c.ret();
    c.label(0);
    c.ret();
    let bytes = c.finish();
    let text = disassemble(&bytes);
    let ok = text.starts_with("jmp") && bytes.len() == 7;
    if !ok {
        failures += 1;
        println!("FAIL  forward jmp: {text}");
    } else {
        println!("ok    forward jmp             {text}");
    }

    println!("\n{} case(s) failed", failures);
    if failures != 0 {
        std::process::exit(1);
    }
}

/// `llvm-mc` separates the mnemonic from its operands with a TAB; comparing raw text would fail on
/// whitespace for every single case. Normalise both sides instead of writing tabs into 35 strings.
fn normalise(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn emit(f: impl FnOnce(&mut Code)) -> Vec<u8> {
    let mut c = Code::new();
    f(&mut c);
    c.finish()
}

/// Assemble one instruction for x86-64 and return its bytes.
fn assemble(asm: &str) -> Vec<u8> {
    let mut child = Command::new("llvm-mc")
        .args(["-triple=x86_64", "-show-encoding"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("llvm-mc on PATH");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(asm.as_bytes())
        .expect("write");
    let out = child.wait_with_output().expect("llvm-mc");
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.stderr.is_empty() {
        println!("      assembler rejected `{asm}`: {}", String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or(""));
        return Vec::new();
    }
    let mut bytes = Vec::new();
    for line in stdout.lines() {
        let Some(start) = line.find("encoding: [") else { continue };
        let end = line[start..].find(']').expect("closing bracket") + start;
        for b in line[start + 11..end].split(',') {
            if let Ok(v) = u8::from_str_radix(b.trim().trim_start_matches("0x"), 16) {
                bytes.push(v);
            }
        }
    }
    bytes
}

fn disassemble(bytes: &[u8]) -> String {
    let text = bytes
        .iter()
        .map(|b| format!("0x{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut child = Command::new("llvm-mc")
        .args(["-triple=x86_64", "-disassemble"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("llvm-mc on PATH");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(text.as_bytes())
        .expect("write");
    let out = child.wait_with_output().expect("llvm-mc");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('.'))
        .collect::<Vec<_>>()
        .join("; ")
}
