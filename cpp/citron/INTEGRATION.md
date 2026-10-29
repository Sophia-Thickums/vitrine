# Wiring this backend into Citron Neo — the exact steps

Written against Citron Neo `c3aa70f`. Nothing here has been compiled inside the emulator; the
adapter itself compiles and passes its checks against the shim (see README.md).

## 1. Add the two files to the emulator

    src/core/arm/vitrine/arm_vitrine_64.h
    src/core/arm/vitrine/arm_vitrine_64.cpp

Change the include of `citron_shim.h` to the real ones:

```cpp
#include "common/common_types.h"
#include "core/arm/arm_interface.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"
```

and change every `void*` thread parameter to `Kernel::KThread*`.

## 2. Replace the modelled memory with Citron's

The shim's `GuestMemory` disappears. The constructor takes `Core::Memory::Memory&` and the reader
becomes:

```cpp
std::int64_t ArmVitrine64::ReadWordThunk(void* ctx, std::uint64_t addr) {
    auto* self = static_cast<ArmVitrine64*>(ctx);
    u32 word = 0;
    if (!self->m_memory.Read32(addr, word)) {
        return -1;                      // out of band, never a sentinel word
    }
    return static_cast<std::int64_t>(word);
}
```

`Memory::Read32` already returns false for unmapped pages, which is the contract the translator
wants. Do **not** substitute a zero word: every 32-bit pattern is a legal instruction.

## 3. Build the Rust side and link it

```cmake
add_library(vitrine STATIC IMPORTED)
set_target_properties(vitrine PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/externals/vitrine/lib/libvitrine.a)
target_include_directories(<citron-core> PRIVATE ${CMAKE_SOURCE_DIR}/externals/vitrine/include)
target_link_libraries(<citron-core> PRIVATE vitrine)
```

`cargo build --release` emits both `libvitrine.a` and `libvitrine.so`; the static one is what the
README recommends for a fork.

## 4. Construct it where the Dynarmic backend is constructed

`ArmDynarmic64` is built per process with a `DynarmicExclusiveMonitor`. Put the vitrine backend
behind a setting so a fork can compare them, and keep Dynarmic as the default until the gaps listed
in README.md close.

## 5. Save and restore the thread context around a run

The shim passes `void*`; the real `RunThread(Kernel::KThread*)` must:

1. `SetContext(thread->GetContext())` before the first run,
2. run until a halt reason,
3. `GetContext(...)` and write it back to the thread,
4. return the reason.

Until step 5 exists, the guest runs with whatever context the last caller left in the state block —
which is the kind of fault that looks like a random register corruption three calls later.

## 6. What must be added before this can boot a game

- **LL/SC** (`ldxr`/`stxr`/`ldaxr`/`stlxr`): the exclusive monitor is not wired at all.
- **SIMD/NEON**: undecoded in the translator, and real Switch code uses it heavily.
- **DataAbort raising**: the backend stops and reports; mapping the page and raising the guest
  exception is the kernel side and is unwritten.
- **AArch32**: stays Dynarmic.
