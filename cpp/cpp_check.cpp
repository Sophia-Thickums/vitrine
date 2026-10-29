// Minimal C++ caller — proves the boundary the way Citron will use it.
//
// The structs come from vitrine.h rather than being redeclared here, so that a mismatch between
// the header and the Rust ABI is a compile error or a wrong answer in CI, not a crash on a user's
// machine. This is the same call sequence `shader_environment.cpp` will make.
//
// Build:  see build_cpp.sh
// Run:    ./cpp_check <path-to-cache-file>

#include <cstdio>
#include <cstdlib>

#include "vitrine.h"

int main(int argc, char** argv) {
    std::printf("vitrine version: %s\n", vitrine_version());

    if (argc < 2) {
        std::printf("usage: %s <cache-file>\n", argv[0]);
        return 2;
    }

    Report report = vitrine_cache_inspect(argv[1]);
    std::printf("status=%d records=%zu envs=%zu consumed=%zu damage_at=%zu\n",
                report.status, report.record_count, report.env_count,
                report.bytes_consumed, report.damage_offset);

    if (report.status == VITRINE_WARN_DAMAGED) {
        std::printf("cache is damaged; repairing by truncation\n");
        int written = vitrine_cache_repair(argv[1], &report, 1);
        std::printf("repair wrote %d record(s)\n", written);
        if (written < 0) {
            std::printf("FAIL: repair returned %d\n", written);
            vitrine_report_free(&report);
            return 1;
        }
    }
    vitrine_report_free(&report);

    // Append one pipeline, exactly as the renderer does after compiling a shader.
    static const unsigned long long code[2] = {0xd503201fULL, 0xd503201fULL};
    CEntry entry{};
    entry.start_address = 0x200;
    entry.viewport_transform_state = 1;
    entry.stage = 4; // Fragment
    entry.code = code;
    entry.code_word_count = 2;

    PendingPipeline pipeline{};
    for (int i = 0; i < 16; ++i) {
        pipeline.key[i] = static_cast<unsigned char>(i + 1);
    }
    pipeline.envs = &entry;
    pipeline.env_count = 1;

    int appended = vitrine_cache_append(argv[1], &pipeline);
    std::printf("append returned %d\n", appended);
    if (appended != 1) {
        std::printf("FAIL: append\n");
        return 1;
    }

    Report after = vitrine_cache_inspect(argv[1]);
    std::printf("after append: status=%d records=%zu envs=%zu\n",
                after.status, after.record_count, after.env_count);
    vitrine_report_free(&after);

    if (after.status != VITRINE_OK) {
        std::printf("FAIL: cache not clean after append (status %d)\n", after.status);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
