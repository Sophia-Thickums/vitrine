#include "arm_vitrine_64.h"

#include <cstring>

namespace Core {

namespace {

// PSTATE bits used by the guest: N=31, Z=30, C=29, V=28. The JIT packs them into one byte as
// bit3=N, bit2=Z, bit1=C, bit0=V, because generated code reads that byte as its single source of
// truth for condition evaluation.
constexpr u32 kPstateN = 1u << 31;
constexpr u32 kPstateZ = 1u << 30;
constexpr u32 kPstateC = 1u << 29;
constexpr u32 kPstateV = 1u << 28;

u8 PackFlags(u32 pstate) {
    u8 f = 0;
    if (pstate & kPstateN) f |= 0b1000;
    if (pstate & kPstateZ) f |= 0b0100;
    if (pstate & kPstateC) f |= 0b0010;
    if (pstate & kPstateV) f |= 0b0001;
    return f;
}

u32 UnpackFlags(u8 flags, u32 rest) {
    u32 p = rest & ~(kPstateN | kPstateZ | kPstateC | kPstateV);
    if (flags & 0b1000) p |= kPstateN;
    if (flags & 0b0100) p |= kPstateZ;
    if (flags & 0b0010) p |= kPstateC;
    if (flags & 0b0001) p |= kPstateV;
    return p;
}

/// SVC #imm16 occupies bits 20..5 of the word, with 0xD4000001 in the rest.
u32 SvcImmediate(u32 word) {
    return (word >> 5) & 0xFFFFu;
}

} // namespace

ArmVitrine64::ArmVitrine64(const GuestMemory& memory, bool uses_wall_clock)
    : ArmInterface{uses_wall_clock}, m_memory{memory} {
    m_jit = vitrine_jit_new_default(&ArmVitrine64::ReadWordThunk, this);
}

ArmVitrine64::~ArmVitrine64() {
    vitrine_jit_free(m_jit);
    m_jit = nullptr;
}

std::int64_t ArmVitrine64::ReadWordThunk(void* ctx, std::uint64_t addr) {
    auto* self = static_cast<ArmVitrine64*>(ctx);
    u32 word = 0;
    if (!self->m_memory.Read32(addr, word)) {
        // Out of band, never a sentinel word: every 32-bit pattern is a legal instruction.
        return -1;
    }
    return static_cast<std::int64_t>(word);
}

HaltReason ArmVitrine64::RunThread(void* /*thread*/) {
    if (m_jit == nullptr) {
        return HaltReason::BreakLoop;
    }
    for (;;) {
        // exchange, not load: observing the flag and clearing it are one operation, so a signal
        // that arrives between the read and the clear cannot be eaten. Clearing it on ENTRY (the
        // first version) discarded any interrupt raised before RunThread was called, which is the
        // common case for a stop requested by another thread.
        if (m_interrupted.exchange(false, std::memory_order_acq_rel)) {
            return HaltReason::BreakLoop;
        }
        const VitrineRunResult r = vitrine_jit_run(m_jit, kBlocksPerCheck);
        if (r.status != VITRINE_OK) {
            // A refusal is a stop, not a crash: an untranslatable instruction or an unmapped fetch
            // is reported as a break so the kernel can decide what the guest sees.
            return HaltReason::BreakLoop;
        }
        switch (r.exit_code) {
        case VITRINE_EXIT_SVC:
            return HaltReason::SupervisorCall;
        case VITRINE_EXIT_DATA_ABORT:
            // pc is left on the faulting instruction, so mapping the page and re-entering re-runs it.
            return HaltReason::DataAbort;
        case VITRINE_EXIT_NEXT:
        case VITRINE_EXIT_JUMPED:
            if (r.executed == 0) {
                return HaltReason::BreakLoop;
            }
            break;
        default:
            return HaltReason::BreakLoop;
        }
    }
}

HaltReason ArmVitrine64::StepThread(void* /*thread*/) {
    if (m_jit == nullptr) {
        return HaltReason::StepThread;
    }
    const VitrineRunResult r = vitrine_jit_step(m_jit);
    if (r.status != VITRINE_OK) {
        return HaltReason::BreakLoop;
    }
    if (r.exit_code == VITRINE_EXIT_SVC) {
        return HaltReason::SupervisorCall;
    }
    if (r.exit_code == VITRINE_EXIT_DATA_ABORT) {
        return HaltReason::DataAbort;
    }
    return HaltReason::StepThread;
}

void ArmVitrine64::ClearInstructionCache() {
    if (m_jit != nullptr) {
        vitrine_jit_invalidate(m_jit, 0, ~std::uint64_t{0});
    }
}

void ArmVitrine64::InvalidateCacheRange(u64 addr, std::size_t size) {
    if (m_jit != nullptr) {
        vitrine_jit_invalidate(m_jit, addr, size);
    }
}

void ArmVitrine64::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    if (m_jit == nullptr) {
        return;
    }
    for (u32 i = 0; i < 29; ++i) {
        ctx.r[i] = vitrine_jit_get_reg(m_jit, i);
    }
    ctx.fp = vitrine_jit_get_reg(m_jit, 29);
    ctx.lr = vitrine_jit_get_reg(m_jit, 30);
    ctx.sp = vitrine_jit_get_sp(m_jit);
    ctx.pc = vitrine_jit_get_pc(m_jit);
    ctx.pstate = UnpackFlags(vitrine_jit_get_flags(m_jit), ctx.pstate);
    ctx.tpidr = m_tpidrro_el0;
}

void ArmVitrine64::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    if (m_jit == nullptr) {
        return;
    }
    for (u32 i = 0; i < 29; ++i) {
        vitrine_jit_set_reg(m_jit, i, ctx.r[i]);
    }
    vitrine_jit_set_reg(m_jit, 29, ctx.fp);
    vitrine_jit_set_reg(m_jit, 30, ctx.lr);
    vitrine_jit_set_sp(m_jit, ctx.sp);
    vitrine_jit_set_pc(m_jit, ctx.pc);
    vitrine_jit_set_flags(m_jit, PackFlags(ctx.pstate));
    m_tpidrro_el0 = ctx.tpidr;
}

void ArmVitrine64::SetTpidrroEl0(u64 value) {
    // Held here, not in the translation. Generated code does not read TPIDRRO_EL0 yet, and a field
    // that no emitted instruction consumes would be a lie about what is implemented.
    m_tpidrro_el0 = value;
}

void ArmVitrine64::GetSvcArguments(std::span<u64, 8> args) const {
    if (m_jit == nullptr) {
        return;
    }
    for (u32 i = 0; i < 8; ++i) {
        args[i] = vitrine_jit_get_reg(m_jit, i);
    }
}

void ArmVitrine64::SetSvcArguments(std::span<const u64, 8> args) {
    if (m_jit == nullptr) {
        return;
    }
    for (u32 i = 0; i < 8; ++i) {
        vitrine_jit_set_reg(m_jit, i, args[i]);
    }
}

u32 ArmVitrine64::GetSvcNumber() const {
    if (m_jit == nullptr) {
        return 0;
    }
    // The run stops WITH pc on the SVC, so the number is decoded from the instruction there rather
    // than tracked in a field that a block could overwrite.
    u32 word = 0;
    if (!m_memory.Read32(vitrine_jit_get_pc(m_jit), word)) {
        return 0;
    }
    return SvcImmediate(word);
}

void ArmVitrine64::SignalInterrupt(void* /*thread*/) {
    m_interrupted.store(true, std::memory_order_relaxed);
}

} // namespace Core
