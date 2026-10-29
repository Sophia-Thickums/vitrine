# The Citron Neo backend — what is here, and what is not proven

`arm_vitrine_64.{h,cpp}` is a `Core::ArmInterface` implementation that runs guest code through
libvitrine instead of Dynarmic.

## The shim, and why it exists

`shim/citron_shim.h` is a **copy of declarations** from Citron Neo at commit `c3aa70f`:

| Citron file | Copied |
|---|---|
| `src/core/arm/arm_interface.h` | `Core::HaltReason` (values verbatim), `Core::ArmInterface` |
| `src/core/hle/kernel/svc_types.h` | `Kernel::Svc::ThreadContext` (layout pinned by `static_assert(sizeof == 0x320)`) |

Nothing else is copied. The point is to compile and TEST the adapter without a full Citron build
(Qt, Boost, the dynarmic submodule, the whole emulator), while keeping the declarations identical so
a drift is a compile error rather than a wrong answer.

## What the check proves

`./cpp/citron/build_adapter_check.sh && ./cpp/citron/adapter_check` — **18 checks**, all through the
`ArmInterface` base class the kernel calls:

- architecture, context round-trip including the packed N/Z/C/V flags
- `StepThread` advancing exactly one instruction, not one block
- `RunThread` stopping **on** the SVC with `pc` on it and the immediate decodable (`svc #0x42`)
- SVC arguments read and written back
- an unmapped load reported as `DataAbort`, `pc` left on the faulting instruction for retry
- a full instruction-cache clear followed by a correct run
- `SignalInterrupt` stopping the run

Two defects were found by these checks and fixed here rather than in the emulator:

1. `RunThread` cleared the interrupt flag on entry, so a signal raised **before** the run was
   discarded — the common case for a stop requested by another thread. It is now an `exchange` on
   observation, so raising and servicing cannot interleave into a lost interrupt.
2. `StepThread` was originally block-granular. The Rust side gained `AddressJit::step`, which
   translates one instruction **uncached** — a cached one-instruction block at the same address
   would be used by the next `RunThread` and the guest would execute one instruction and stop.

## What is NOT proven, said plainly

- **It has never been compiled inside Citron.** The shim is faithful to the commit named above, but
  the real build brings headers this shim does not model (KThread, KProcess, watchpoints, the
  exclusive monitor). Expect a handful of signature fixes on the first real compile.
- **No thread context yet.** `RunThread` and `StepThread` take `void*` here; in the patch they take
  `Kernel::KThread*` and must save/restore the context around the run.
- **No LL/SC.** The exclusive monitor is not wired; `ldxr`/`stxr` are not translated yet.
- **`DataAbort` stops; it does not raise.** Mapping the page and raising the guest exception is the
  kernel's side of that call and is not written.
- **AArch32 is not implemented.** `GetArchitecture()` is constant and Citron's 32-bit backend stays
  Dynarmic.
- **TPIDRRO_EL0 is held out of band** — generated code does not read it yet, and a field no emitted
  instruction consumes would be a claim about what is implemented.

## ★ THE ADAPTER HAS NO DATA WINDOW YET (2026-10-29)

`ArmVitrine64` builds its state with `vitrine_jit_new_default()`, which leaves `mem_ptr = 0` and
`mem_len = 0`. The generated guard therefore refuses **every** guest load and store with `DataAbort`,
which is the safe failure — it cannot read or write host memory — but it also means no guest data access
works, so this backend cannot yet run a program that touches memory.

Wiring it needs one decision, and it is Citron's to make because its RAM is a sparse page table with two
attribute bits packed into each entry: either pass a real flat window covering the pages in play, or move
data access onto the same fallible reader the instruction fetch already uses. **A fabricated `mem_ptr` is
not an error the guard can catch** — the bounds check proves an address is inside the window, not that the
window is host memory, and a test here segfaulted doing exactly that with a base of 0.

Also note `RunThread` currently checks `r.status` and maps a refusal to `BreakLoop`; once a window exists,
a guest `DataAbort` should reach the kernel as `HaltReason::DataAbort` so the page can be mapped and the
instruction retried.
