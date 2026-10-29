// The JIT half of the boundary: a C++ caller driving the translator the way Citron's ArmInterface
// backend will. Everything here goes through vitrine.h — no struct is redeclared locally, so a
// header/ABI mismatch is a compile error or a wrong answer rather than a crash on a user's machine.
//
// Build: ./cpp/build_jit_check.sh     Run: ./cpp/jit_check

#include <cstdio>
#include <cstdint>
#include <vector>

#include "vitrine.h"

namespace {

// One readable page at address 0 and nothing anywhere else. The reader refusing is the point: a
// sentinel word would be a legal instruction, so unmapped has to be out-of-band.
struct Guest {
    std::vector<uint32_t> code;
    int64_t reads = 0;
};

int64_t read_word(void* ctx, uint64_t addr) {
    auto* g = static_cast<Guest*>(ctx);
    g->reads++;
    if (addr % 4 != 0) return -1;
    const size_t i = addr / 4;
    if (i >= g->code.size()) return -1;
    return static_cast<int64_t>(g->code[i]);
}

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
    else     { std::printf("ok   %s\n", what); }
}

} // namespace

int main() {
    // mov x0, #1 ; add x1, x0, #2 ; ret
    Guest guest{{0xD2800020u, 0x91000801u, 0xD65F03C0u}};

    auto* memory = new Guest{guest};
    VitrineJit* jit = vitrine_jit_new_default(read_word, memory);
    check(jit != nullptr, "handle created through the C ABI");

    // Return to an address with nothing mapped, so the run stops cleanly after one block.
    vitrine_jit_set_reg(jit, 30, 0x10000);
    vitrine_jit_set_pc(jit, 0);

    VitrineRunResult r = vitrine_jit_run(jit, 1);
    check(r.status == 0, "run completed");
    check(r.executed == 1, "one block retired");
    check(vitrine_jit_get_reg(jit, 1) == 3, "x1 = x0 + 2 = 3");
    check(vitrine_jit_get_pc(jit) == 0x10000, "pc followed the return");
    check(r.exit_code == VITRINE_EXIT_JUMPED, "exit code says it jumped");

    check(vitrine_jit_cache_len(jit) == 1, "one translation cached");
    check(vitrine_jit_invalidate(jit, 8, 4) == 1,
          "invalidating the block's TAIL drops it (overlap, not containment)");
    check(vitrine_jit_cache_len(jit) == 0, "cache empty after invalidation");

    // An unmapped fetch must refuse rather than execute zeros.
    vitrine_jit_set_pc(jit, 0x8000);
    VitrineRunResult miss = vitrine_jit_run(jit, 1);
    check(miss.status != 0, "fetching from an unmapped address is refused");
    check(miss.executed == 0, "and nothing was executed");

    vitrine_jit_free(jit);
    delete memory;

    std::printf(failures == 0 ? "ALL CHECKS PASSED\n" : "%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
