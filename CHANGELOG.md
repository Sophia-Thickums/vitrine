# Changelog

All three releases are on GitHub: https://github.com/Sophia-Thickums/vitrine/releases

## 0.1.2 — the ship pass

Nothing in the translator changed. What changed is everything a fork needs.

- **CI** (`.github/workflows/ci.yml`): format, clippy with `-D warnings`, the full suite, both C++
  programs built **and run**, the CMake package config exercised from an outside project, and a
  tag/version guard.
- **`./build.sh`** — one command running exactly what CI runs, in the same order.
- **`CONTRIBUTING.md`** — the rules that came from being wrong: generate encodings with `llvm-mc`,
  every fix ships an arm that fails without it, never weaken a check to go green.
- **`rust-toolchain.toml`** — stable with rustfmt and clippy, so a fork does not build on whatever
  happens to be installed.
- **Version truth.** v0.1.1 shipped with a crate that still reported `0.1.0` through
  `vitrine_version()`. The version now matches the tag, and CI fails a tagged build whose Cargo
  version disagrees with it.
- `Cargo.toml` gains repository, readme, keywords and categories.

## 0.1.1 — CMake package config

- `find_package(vitrine 0.1 CONFIG REQUIRED)` resolves `vitrine::vitrine` from
  `lib/libvitrine.a` + `include/vitrine.h`, with the `Threads`/`dl`/`m` link interface a Rust
  staticlib needs. Verified by an outside CMake project given only `CMAKE_PREFIX_PATH`.
- Library code unchanged. **This release reports crate version 0.1.0 — see 0.1.2.**

## 0.1.0 — first release

Three jobs, each with a check that can fail.

- **Vulkan pipeline cache**: read, validate, repair, append. Byte-compatible with the C++ format
  (`yuzuvkch`, version 1, 16-byte keys), with deterministic serialisation via `BTreeMap`.
- **Content selection**: the update and DLC that actually load, chosen by title version instead of by
  whichever CNMT the directory walk reached last.
- **AArch64 translator**: decoder, basic-block analysis (whole-region and at-an-address), a reference
  interpreter pinned to `qemu-aarch64`, an x86-64 encoder, an address-keyed block cache, a
  guest-memory guard that aborts instead of reading host memory, and a C ABI.
- **`Core::ArmInterface` backend** (`cpp/citron/`), compiled and tested against a faithful shim of
  Citron's own headers at `c3aa70f`.

Measured at release: 127 Rust tests, clippy clean, 18/18 programs identical to `qemu-aarch64`, 47/47
x86 encodings identical to `llvm-mc`, 11 checks in the C++ JIT boundary, 18 in the adapter.

Limits, stated in the README rather than implied: no SIMD/NEON, no LL/SC, no AArch32, no MMU in the
JIT's memory model, no HLE services, and the backend has never been compiled inside Citron.
