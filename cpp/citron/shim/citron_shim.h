// A FAITHFUL, MINIMAL SHIM of the Citron Neo interfaces the vitrine backend implements.
//
// WHY THIS FILE EXISTS: the backend must be compiled and tested somewhere, and a full Citron build
// (Qt, Boost, dynarmic submodule, the whole emulator) is not something this repo can do. So the
// declarations below are copied from Citron's own headers at commit c3aa70f and nothing else is:
//
//   src/core/arm/arm_interface.h          -> Core::HaltReason, Core::ArmInterface
//   src/core/hle/kernel/svc_types.h       -> Kernel::Svc::ThreadContext
//
// They are transcribed exactly (field names, order, widths) so that a drift in either direction is
// a compile error here rather than a wrong answer inside the emulator. If Citron's header changes,
// this shim must change with it — that is the point of keeping the copy small and naming the commit.
//
// What is NOT here: KThread, KProcess, watchpoints, the exclusive monitor. The backend reaches those
// through Citron's own types in the real patch; the shim only supplies what the class signature
// needs and models memory as one virtual read.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Core {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

/// Matches Citron's HaltReason exactly — the numeric values are what the kernel switches on.
enum class HaltReason : u64 {
    StepThread = 0x00000001,
    DataAbort = 0x00000004,
    BreakLoop = 0x02000000,
    SupervisorCall = 0x04000000,
    InstructionBreakpoint = 0x08000000,
    PrefetchAbort = 0x20000000,
};

constexpr HaltReason operator|(HaltReason a, HaltReason b) {
    return static_cast<HaltReason>(static_cast<u64>(a) | static_cast<u64>(b));
}
constexpr HaltReason operator&(HaltReason a, HaltReason b) {
    return static_cast<HaltReason>(static_cast<u64>(a) & static_cast<u64>(b));
}

enum class Architecture { AArch64, AArch32 };

namespace Kernel::Svc {

/// Copied from svc_types.h:461. 0x320 bytes, and the static_assert below keeps it that way.
struct ThreadContext {
    std::array<u64, 29> r;  // x0 .. x28
    u64 fp;                 // x29
    u64 lr;                 // x30
    u64 sp;
    u64 pc;
    u32 pstate;
    u32 padding;
    std::array<u128, 32> v;
    u32 fpcr;
    u32 fpsr;
    u64 tpidr;
};
static_assert(sizeof(ThreadContext) == 0x320, "ThreadContext layout drifted from Citron's");

} // namespace Kernel::Svc

/// The memory surface the backend needs: one word read that can fail.
///
/// In the real patch this is `system.Memory().Read32(addr, &out)` plus the page-table type check;
/// here it is virtual so a test can supply a guest with holes in it.
class GuestMemory {
public:
    virtual ~GuestMemory() = default;
    /// Returns false when `addr` is not mapped — the backend must never turn that into a zero word.
    virtual bool Read32(u64 addr, u32& out) const = 0;
};

/// The subset of Citron's ArmInterface this backend implements.
///
/// Signatures copied from arm_interface.h; the virtuals are pure here exactly as they are there, so
/// forgetting to implement one is a compile error.
class ArmInterface {
public:
    explicit ArmInterface(bool uses_wall_clock) : m_uses_wall_clock{uses_wall_clock} {}
    virtual ~ArmInterface() = default;

    virtual void Initialize() {}

    virtual HaltReason RunThread(void* thread) = 0;
    virtual HaltReason StepThread(void* thread) = 0;

    virtual void LockThread(void* /*thread*/) {}
    virtual void UnlockThread(void* /*thread*/) {}

    virtual void ClearInstructionCache() = 0;
    virtual void InvalidateCacheRange(u64 addr, std::size_t size) = 0;

    virtual Architecture GetArchitecture() const = 0;

    virtual void GetContext(Kernel::Svc::ThreadContext& ctx) const = 0;
    virtual void SetContext(const Kernel::Svc::ThreadContext& ctx) = 0;
    virtual void SetTpidrroEl0(u64 value) = 0;

    virtual void GetSvcArguments(std::span<u64, 8> args) const = 0;
    virtual void SetSvcArguments(std::span<const u64, 8> args) = 0;
    virtual u32 GetSvcNumber() const = 0;

    virtual void SignalInterrupt(void* thread) = 0;

protected:
    bool m_uses_wall_clock{};
};

} // namespace Core
