// The vitrine backend for Citron Neo's Arm64 CPU.
//
// One class, one job: satisfy Core::ArmInterface by translating guest blocks through libvitrine and
// executing them on the host, instead of through Dynarmic. Everything the kernel calls lands here.
//
// Scope, stated plainly:
//   * AArch64 only. GetArchitecture() is constant and AArch32 is not implemented — Citron's 32-bit
//     backend is a separate class and this one does not pretend to cover it.
//   * Blocks are translated from guest memory through a fallible reader, so an unmapped fetch and an
//     unmapped load are both refused rather than executed over whatever the host happens to hold.
//   * StepThread is instruction-granular (a one-instruction translation, compiled uncached), which
//     is slow and correct — the right trade for a debug path.
#pragma once

#include <atomic>
#include <cstdint>
#include <span>

#include "citron_shim.h"
#include "vitrine.h"

namespace Core {

class ArmVitrine64 final : public ArmInterface {
public:
    /// `memory` must outlive the CPU. `uses_wall_clock` mirrors the interface's own flag.
    explicit ArmVitrine64(const GuestMemory& memory, bool uses_wall_clock);
    ~ArmVitrine64() override;

    ArmVitrine64(const ArmVitrine64&) = delete;
    ArmVitrine64& operator=(const ArmVitrine64&) = delete;

    Architecture GetArchitecture() const override {
        return Architecture::AArch64;
    }

    HaltReason RunThread(void* thread) override;
    HaltReason StepThread(void* thread) override;

    void ClearInstructionCache() override;
    void InvalidateCacheRange(u64 addr, std::size_t size) override;

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override;
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override;
    void SetTpidrroEl0(u64 value) override;

    void GetSvcArguments(std::span<u64, 8> args) const override;
    void SetSvcArguments(std::span<const u64, 8> args) override;
    u32 GetSvcNumber() const override;

    void SignalInterrupt(void* thread) override;

private:
    /// Trampoline for the C ABI: the reader has no user-data slot of its own beyond `ctx`.
    static std::int64_t ReadWordThunk(void* ctx, std::uint64_t addr);

    /// Blocks retired between interrupt checks. Small enough that a guest thread yields often,
    /// large enough that the check is not the cost of the run.
    static constexpr u64 kBlocksPerCheck = 64;

    const GuestMemory& m_memory;
    /// Set by SignalInterrupt from another thread; read by the run loop between checks.
    std::atomic<bool> m_interrupted{};
    VitrineJit* m_jit{};
    u64 m_tpidrro_el0{};
};

} // namespace Core
