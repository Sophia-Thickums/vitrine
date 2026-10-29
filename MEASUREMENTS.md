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
