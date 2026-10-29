# Contributing

This crate exists to be forked into Citron Neo. A change here is worth making if it either fixes
something measured or adds a check that can fail.

## The one command

```sh
./build.sh
```

It runs exactly what CI runs, in the same order: `cargo fmt --check`, clippy with warnings denied,
the full test suite, both C++ programs **built and executed**, and the CMake package config
exercised from an outside project. If it passes locally and fails on GitHub, that is a bug in
`build.sh` and it matters more than whatever you were changing.

## Rules that came from being wrong

**Generate encodings, never type them.** Every instruction word in `tests/fixtures/*.tsv` was
produced by `llvm-mc`. Five hand-typed encodings were wrong in the first draft of the interpreter
tests, and a wrong encoding makes a test that passes for the wrong reason.

```sh
echo "ldp x0, x1, [x2]" | llvm-mc -triple=aarch64 -show-encoding
```

**A fix ships with an arm that fails without it.** Not a test that passes after the change — a test
that goes red when the change is reverted. `git stash` the fix, run it, watch it fail, restore.
Several tests here were confirmed this way, and the commit messages say so.

**Prove the harness reaches the branch.** A test can pass because an earlier refusal answered first.
If a new check cannot print its own verdict on the input you intended, it is not testing what you
think.

**A check that only prints is not a check.** It must exit non-zero. The empty-string hash
(`e3b0c442…`) is what a "verification" produces when it read nothing.

**Do not weaken a check to make it green.** If a bound is wrong, fix the bound; if the tolerance is
wrong, say why in the commit. Deleting an assertion to land a change is how a suite becomes theatre.

**Measurements go in `MEASUREMENTS.md`, with the method.** Numbers in a commit message age; numbers
in a file with the instrument that produced them can be re-run.

## Where a change belongs

| change | file |
|---|---|
| a decoded instruction | `src/a64.rs` + a fixture row + an interpreter test |
| a compiled instruction | `src/x86.rs` + a `tests/jit.rs` case comparing against the interpreter |
| the C boundary | `src/ffi.rs` + `include/vitrine.h` + a test in `tests/ffi.rs` |
| the Citron backend | `cpp/citron/` — and `INTEGRATION.md` if the shape changes |
| a performance claim | `examples/` + a row in `MEASUREMENTS.md` |

## What is not wanted

- **Nintendo material of any kind**: keys, firmware, ROMs, title dumps, decryption tooling. This
  crate is an ARM translator and a cache reader. That is why it can be public.
- **Household detail in public text.** No hostnames, paths or operator names in committed files.
- **Silent fallbacks.** An instruction that is not implemented returns `Err` and is visible. A
  decoder that guesses is worse than one that refuses.

## Licence

Contributions are GPL-3.0-or-later, matching Citron Neo. By opening a pull request you agree to that.
