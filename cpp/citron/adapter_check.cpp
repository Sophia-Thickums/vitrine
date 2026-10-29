// Drives the vitrine backend the way Citron's kernel will: through Core::ArmInterface, with the
// thread context, the SVC path and the fault path all exercised.
//
// This compiles against the SHIM (cpp/citron/shim/citron_shim.h), which is a faithful copy of the
// declarations in Citron commit c3aa70f, not against a Citron build. That is the honest limit of
// what can be proven here: the adapter's logic and its ABI use are tested; that it links inside the
// emulator is a separate step that needs the emulator.
//
// Build: ./cpp/citron/build_adapter_check.sh      Run: ./cpp/citron/adapter_check

#include <cstdio>
#include <cstring>
#include <vector>

#include "arm_vitrine_64.h"

namespace {

using Core::u32;
using Core::u64;

// A guest with one mapped page at 0 and a hole everywhere else, which is the smallest thing that
// can tell a working fault path from a reader that returns zeros.
class FakeMemory final : public Core::GuestMemory {
public:
    explicit FakeMemory(std::vector<u32> code) : m_code{std::move(code)} {}

    bool Read32(u64 addr, u32& out) const override {
        if (addr % 4 != 0) {
            return false;
        }
        const size_t i = addr / 4;
        if (i >= m_code.size()) {
            return false;
        }
        out = m_code[i];
        return true;
    }

private:
    std::vector<u32> m_code;
};

int g_failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++g_failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

} // namespace

int main() {
    // mov x0, #1 ; add x1, x0, #2 ; svc #0x42 ; ret
    FakeMemory memory(std::vector<u32>{0xD2800020u, 0x91000801u, 0xD4000841u, 0xD65F03C0u});
    Core::ArmVitrine64 cpu{memory, false};

    Check(cpu.GetArchitecture() == Core::Architecture::AArch64, "architecture is AArch64");

    Core::Kernel::Svc::ThreadContext ctx{};
    ctx.pc = 0;
    ctx.sp = 0x7FF000;
    ctx.pstate = 0x40000000u; // Z set, everything else clear
    cpu.SetContext(ctx);

    Core::Kernel::Svc::ThreadContext readback{};
    cpu.GetContext(readback);
    Check(readback.pc == 0, "pc round-trips through the context");
    Check(readback.sp == 0x7FF000, "sp round-trips");
    Check((readback.pstate & 0x40000000u) != 0, "the Z flag round-trips through the packed byte");
    Check((readback.pstate & 0x80000000u) == 0, "and N stays clear");

    // StepThread must advance exactly one instruction and say so.
    const Core::HaltReason stepped = cpu.StepThread(nullptr);
    cpu.GetContext(readback);
    Check(stepped == Core::HaltReason::StepThread, "one step reports StepThread");
    Check(readback.pc == 4, "one step advanced pc by exactly 4");
    Check(readback.r[0] == 1, "and the mov executed: x0 = 1");

    // RunThread must stop ON the svc, with the number readable from the instruction there.
    const Core::HaltReason halted = cpu.RunThread(nullptr);
    Check(halted == Core::HaltReason::SupervisorCall, "RunThread stops at the svc");
    cpu.GetContext(readback);
    Check(readback.pc == 8, "pc is ON the svc, not past it");
    Check(cpu.GetSvcNumber() == 0x42, "the svc immediate decodes to 0x42");
    Check(readback.r[1] == 3, "the add before it ran: x1 = x0 + 2");
    std::array<u64, 8> args_storage{};
    std::span<u64, 8> args{args_storage};
    cpu.GetSvcArguments(args);
    Check(args[0] == 1, "svc argument 0 is x0");

    // SVC arguments are writable, which is how the kernel returns a value to the guest.
    std::array<u64, 8> result{};
    result[0] = 0xDEAD;
    cpu.SetSvcArguments(std::span<const u64, 8>{result});
    cpu.GetSvcArguments(args);
    Check(args[0] == 0xDEAD, "SetSvcArguments writes x0 back");

    // A load from an unmapped address is a DataAbort with pc left on the instruction.
    FakeMemory hole(std::vector<u32>{0xF9400020u, 0xD65F03C0u}); // ldr x0, [x1] ; ret
    Core::ArmVitrine64 faulting{hole, false};
    Core::Kernel::Svc::ThreadContext fctx{};
    fctx.pc = 0;
    fctx.r[1] = 0x900000; // nothing mapped there
    faulting.SetContext(fctx);
    Check(faulting.RunThread(nullptr) == Core::HaltReason::DataAbort, "an unmapped load is a DataAbort");
    Core::Kernel::Svc::ThreadContext after{};
    faulting.GetContext(after);
    Check(after.pc == 0, "pc is left on the faulting instruction so it can be retried");

    // Clearing the cache must not corrupt the CPU: the program runs again afterwards.
    cpu.ClearInstructionCache();
    cpu.SetContext(ctx);
    Check(cpu.RunThread(nullptr) == Core::HaltReason::SupervisorCall,
          "runs correctly after a full instruction-cache clear");

    // SignalInterrupt is checked between block batches; a set flag stops the run.
    cpu.SetContext(ctx);
    cpu.SignalInterrupt(nullptr);
    Check(cpu.RunThread(nullptr) == Core::HaltReason::BreakLoop,
          "a signalled interrupt stops the run at the next check");

    std::printf(g_failures == 0 ? "ALL CHECKS PASSED\n" : "%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
