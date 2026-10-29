# Vitrine

The Rust core for [Citron Neo](https://github.com/citron-neo/emulator). It replaces the parts of the
emulator where a memory mistake becomes a crash three sessions later, and hands them back to the
C++ engine across a narrow `extern "C"` boundary.

**It is a component library, not a replacement for the emulator.** The C++ tree keeps its renderer,
its UI, its JIT and its service layer. What moves here is the pipeline cache.

## The bug this exists to fix

Citron Neo's `LoadPipelines` in `src/video_core/shader_environment.cpp` is wrapped in a single
`try` block whose `catch` deletes the entire pipeline cache file. One corrupt record — a power loss
mid-write, a crashed run — and the user's whole shader cache is gone, silently. Worse, the records
read *before* the bad one have already been applied, so the emulator resumes with a half-populated
cache and no file.

Vitrine parses first and never mutates on a parse. Damage is reported with a byte offset. The
caller decides whether to truncate (keep every good record) or discard. Nothing is deleted unless
somebody asks for it.

## Verified behaviour

```
$ ./cpp/cpp_check /tmp/damaged.bin
status=1 records=1 envs=1 consumed=204 damage_at=204
cache is damaged; repairing by truncation
repair wrote 1 record(s)
after append: status=0 records=2 envs=2
OK
select: chosen=2 rejected=2 warnings=0 truncated=0
content selection OK
```

A truncated cache keeps its good records, repairs to a valid file, and accepts a new pipeline.
The C++ code would have deleted all of it.

## What is in the crate

| module | what it is |
|---|---|
| `cache` | read, validate, repair, append the Vulkan pipeline cache |
| `content` | choose the winning update and DLC out of everything installed |
| `env` | the serialized shader environment, field for field with `GenericEnvironment` |
| `cursor` | bounds-checked little-endian reader; every short read is an `Err`, never a panic |
| `ffi` | the C ABI in `include/vitrine.h` |

The cache format is **unchanged**: magic `yuzuvkch`, version 1, 16-byte keys. An existing user's
shaders load without rebuilding. The two implementations can read the same file.

One deliberate difference in the bytes: the C++ writer emits `std::unordered_map` contents in
unspecified order, so the same environment can serialize differently between runs. Vitrine uses
`BTreeMap`, so its output is deterministic and can be checksummed. Readers of either accept both.

### The second defect: which update actually loads

`RegisteredCache::ProcessFiles` fills a `std::map<u64, CNMT>` with

```cpp
out_meta.insert_or_assign(title_id, std::move(cnmt));
```

keyed on the title id, which is identical at every version of a patch. So when two updates for one
game are installed — the ordinary case — **whichever CNMT the directory walk reached last wins**,
and directory order is filesystem order. The game boots on an arbitrary version.

`content::select` takes every CNMT and returns one winner per title id by highest version, with
every loser named and the reason attached. The result is permutation-invariant: shuffle the input
and the same entry wins. Patches and DLC are separate slots from the base game, and an update or
DLC whose base application is missing is retained and flagged rather than silently ignored.


## What is proven, and what is not

Measured, not asserted:

| claim | how it is checked | result |
|---|---|---|
| decoder agrees with the assembler | 46-instruction corpus in `tests/fixtures/a64_corpus.tsv` | all match |
| decoder covers real code | 304 compiler-emitted instructions (clang `-O0/-O2/-Os`) | 100% recognised |
| interpreter agrees with a real emulator | `examples/diff_qemu`: 15 programs through `qemu-aarch64` | 15/15 |
| compiler agrees with the interpreter | `tests/jit.rs`, same programs, both engines | all match |
| x86 encoder agrees with the assembler | `examples/diff_x86`: 38 forms, byte-for-byte | all match |
| the JIT is actually faster | `examples/bench` | 1.87x on a loop; 0.53x on a single-use block |
| a persisted pipeline cache is worth having | `examples/vk_pipeline`, two processes, Mesa cache off | ~80-87x on creation, three AMD cards |

The pipeline measurement, its method, and the reason a driver cannot be used to validate a cache
file are in [`MEASUREMENTS.md`](MEASUREMENTS.md).

**What is NOT proven:** floating point is *decoded* but not *executed* — `FMOV`,
`FMADD`, `FMUL` and FP loads/stores refuse rather than compute, because there is no
FP register file yet. SIMD/NEON, atomics, pointer authentication and the system
registers are not decoded at all. There is no memory protection, no MMU and no TLB:
the interpreter's memory is one flat array. "100% of real instructions recognised" is
a decoding claim about a handful of programs, not a compatibility claim about an ISA.

## Building

```sh
cargo build --release            # libvitrine.so and libvitrine.a
cargo test --release             # 59 tests, zero failures
./cpp/build_cpp.sh               # builds the C++ integration check
./cpp/build_cpp.sh --static      # links the static archive instead
```

Rust 1.85 or newer, edition 2024. No dependencies beyond `thiserror`; the parser uses nothing but
`std`.

## Integrating into Citron Neo

```cmake
# add to the target that already compiles shader_environment.cpp
add_library(vitrine STATIC IMPORTED)
set_target_properties(vitrine PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/externals/vitrine/lib/libvitrine.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/externals/vitrine/include)
target_link_libraries(citron PRIVATE vitrine)
```

Then, in `shader_environment.cpp`, replace the body of `LoadPipelines`:

```cpp
Report report = vitrine_cache_inspect(filename.string().c_str());
if (report.status == VITRINE_WARN_DAMAGED) {
    LOG_WARNING(Common_Filesystem, "pipeline cache damaged at byte {}, keeping {} records",
                report.damage_offset, report.record_count);
    vitrine_cache_repair(filename.string().c_str(), &report, 1);
}
// ... replay report.record_count environments into the live cache ...
vitrine_report_free(&report);
```

Nothing else in the tree changes. To carry the parser inside the emulator binary rather than
linking a shared object, build `target/release/libvitrine.a` (emitted alongside the `.so`) and
link that — `cpp/build_cpp.sh --static` shows the link line.

## What is not here yet, and why

The other two places Rust genuinely pays for itself — the ARM64 JIT and the HLE service layer —
are not in this crate yet. They are larger jobs and they need a booting core to be measured against.
This crate is deliberately the piece that can be finished, tested and merged on its own.

## Status

This is a working core, not a finished emulator. What that means concretely:

- **Decoder** — 100% of instructions in four clang-generated `.text` blobs (`-O0`, `-O2`, `-Os`),
  and 46/46 in a hand-checked corpus. SIMD/NEON, atomics and pointer authentication are **not**
  decoded.
- **Interpreter** — executes integer, branch, load/store, bitfield and scalar-double instructions.
  Agrees with `qemu-aarch64` on 18 programs. Single-precision forms are refused.
- **Compiler** — emits x86-64 for the same set, agrees with the interpreter including the FP
  register file, and is 1.87x faster on a loop. On a straight-line block that runs once it is
  *slower* (0.53x), because the driver's per-entry work dominates.
- **Not present** — MMU/TLB, memory protection, system registers, HLE services, a Vulkan backend,
  thread scheduling. The memory model is one flat array.

## Licence

GPL-3.0-or-later, matching Citron Neo. A GPL project may link this crate. A permissively-licensed
project may not.
