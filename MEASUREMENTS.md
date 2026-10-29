# Pipeline creation cost, measured

Why a pipeline cache exists at all, and what it is worth on this hardware.

## Method

`examples/vk_pipeline` compiles the same 16 compute pipelines twice, in **two separate processes**:

```sh
export MESA_SHADER_CACHE_DISABLE=true      # otherwise Mesa answers and you measure Mesa
vk_pipeline --device <name> --save cache.bin *.spv     # process 1: cold, writes the blob
vk_pipeline --device <name> --load cache.bin *.spv     # process 2: reads the blob
```

Two details decide whether the number is real:

- **The `VkPipelineCache` handle must exist before the cold pass**, or it never sees the
  compilations and the "warm" run is just a second cold one.
- **The two passes must be separate processes.** Measured in one process, radv's in-memory cache
  answers the second pass and the reported speedup is over 500x — true, and meaningless.

`MESA_SHADER_CACHE_DISABLE=true` is not optional either. With Mesa's own disk cache enabled, a
second run of the *whole program* comes back in 0.008 ms per pipeline and looks like a 30x faster
GPU. That is Mesa's cache, not ours.

## Results (16 compute pipelines, one Mesa shader compiled per pipeline)

| device | cold, per pipeline | from a saved cache | ratio |
|---|---|---|---|
| RX 9070 XT (RADV GFX1201) | 0.260 ms | 0.003 ms | ~87x |
| Instinct MI25 (RADV VEGA10) | 0.241 ms | 0.003 ms | ~80x |
| Vega Frontier Edition (RADV VEGA10) | 0.243 ms | 0.003 ms | ~81x |
| cache blob size | 18,832 bytes (RDNA4) / 18,204 bytes (Vega 10) | | |

The blobs are ~18 KB for 16 pipelines — about 1.2 KB per compiled pipeline. That is the object the
repair path in `src/cache.rs` exists to protect.

## The finding that matters for correctness

**A corrupted blob is accepted without an error.** Corrupting the first 16 bytes of a valid cache
and passing it back produces `VK_SUCCESS` and a cache that silently ignores its contents — the
Vulkan specification permits a driver to do exactly that.

So a driver cannot be used as a validator: "the driver accepted it" is not evidence that a cache
file is sound. Structure has to be checked before use, and damage has to be handled as a repair
rather than a refetch. That is the whole reason `LoadPipelines` in the C++ tree must not delete the
file on one bad record, and the reason this crate parses, reports and truncates instead.

## Cross-machine determinism, re-run at HEAD (2026-10-29)

The decoder's determinism claim is not a one-off: re-measured at the commit that added the memory guard
and the address-keyed runtime, on two machines with different CPUs, different glibc and different Rust
builds.

| machine | CPU | words | hash | rate |
|---|---|---|---|---|
| this box | Ryzen 9 9950X3D (Zen 5) | 20,000,000 (seed 0x4660) | **0x5904d5599541f01c** | 54.2 Mw/s |
| RetrOasis | Xeon E5-2697 v2 (Ivy Bridge-EP, 12c/24t) | 20,000,000 (seed 0x4660) | **0x5904d5599541f01c** | 17.6 Mw/s |

Identical checksums AND identical `unknown` counts (14,373,200) on both. A decoder reading uninitialised
memory, relying on unspecified layout, or branching on architecture would not survive this — the two CPUs
are nine years apart and share no AVX-512, so any such dependence shows up as a different hash.

**How to reproduce, and the trap — CORRECTED 2026-10-29 14:10, because the first version of this
note named the wrong cause.** Ship the ordinary `cargo build --release --example shard` binary. That is
all. It runs on the Xeon at 17.5 Mw/s and produces the identical hash above.

What actually happened, in order, since the wrong explanation was committed:

1. A binary built with `RUSTFLAGS=-C target-feature=+crt-static` died on her with `Illegal instruction`,
   and a hello-world built the same way died identically. **That control was read as "this distro's std
   is compiled for a newer microarchitecture". It is not.**
2. A Fedora-container build was launched to get a "generic baseline", its `cargo build --offline`
   **failed** on dependency resolution — and the shell reported `BUILT` anyway, because the pipeline
   ended in `| tail -3` and a pipeline's exit status belongs to its LAST command. The `cp` that followed
   then copied the HOST's binary, which is why the shipped file was byte-identical
   (`sha256 c5f28491…`) to `target/release/examples/shard`.
3. That binary ran on the Xeon without complaint — the same file the container was supposed to replace.

**So the cause was `+crt-static` (a static-pie build), not the CPU baseline, and the fix was to stop
using it.** Two laws with teeth in this: a control tells you *that* something differs, not *what*; and a
pipelines exit status is the last command's, so `cargo build … | tail` can fail and still read as success.
The determinism result above is unaffected — it was produced by that host binary, on both machines, with
the same hash and the same unknown-count.
