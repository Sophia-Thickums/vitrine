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
```

A truncated cache keeps its good records, repairs to a valid file, and accepts a new pipeline.
The C++ code would have deleted all of it.

## What is in the crate

| module | what it is |
|---|---|
| `cache` | read, validate, repair, append the Vulkan pipeline cache |
| `env` | the serialized shader environment, field for field with `GenericEnvironment` |
| `cursor` | bounds-checked little-endian reader; every short read is an `Err`, never a panic |
| `ffi` | the C ABI in `include/vitrine.h` |

The cache format is **unchanged**: magic `yuzuvkch`, version 1, 16-byte keys. An existing user's
shaders load without rebuilding. The two implementations can read the same file.

One deliberate difference in the bytes: the C++ writer emits `std::unordered_map` contents in
unspecified order, so the same environment can serialize differently between runs. Vitrine uses
`BTreeMap`, so its output is deterministic and can be checksummed. Readers of either accept both.

## Building

```sh
cargo build --release            # libvitrine.so and libvitrine.a
cargo test --release             # 34 tests, zero failures
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
are not in this crate. They are larger jobs and they need a booting core to be measured against.
This crate is deliberately the piece that can be finished, tested and merged on its own.

## Licence

GPL-3.0-or-later, matching Citron Neo. A GPL project may link this crate. A permissively-licensed
project may not.
