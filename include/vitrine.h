/* SPDX-FileCopyrightText: Copyright 2026 Sophia Marie DeClue
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * vitrine.h — C interface to the Vitrine Rust core.
 *
 * Drop-in replacement for the pipeline-cache half of `src/video_core/shader_environment.cpp`.
 * The cache file format is unchanged (Yuzu magic `yuzuvkch`, version 1), so an existing user's
 * shaders load without rebuilding.
 *
 * # What changed from the C++ behaviour
 *
 * The C++ `LoadPipelines` deletes the entire cache file when any record fails to parse. This
 * interface instead reports the damage and lets the caller decide:
 *
 *     Report r = vitrine_cache_inspect(cache_path);
 *     if (r.status == VITRINE_WARN_DAMAGED) {
 *         // Keep every record that parsed; lose only the unreadable tail.
 *         vitrine_cache_repair(cache_path, &r, 1);
 *     }
 *     vitrine_report_free(&r);
 *
 * That is the one behavioural difference. Everything else — byte layout, magic, version, key
 * length — is identical, so the two implementations can read the same file.
 *
 * # Thread safety
 *
 * Every function is safe to call from multiple threads with different paths. No global state is
 * kept. A single `Report` must not be shared between threads, and `vitrine_report_free` must be
 * called exactly once per report.
 */

#ifndef VITRINE_H
#define VITRINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- status codes -------------------------------------------------------- */

/** Operation succeeded; the cache is whole. */
#define VITRINE_OK 0
/** Operated, but the cache had damage: records were kept, the unreadable tail was lost. */
#define VITRINE_WARN_DAMAGED 1
/** Caller passed a null pointer or a non-UTF-8 path. */
#define VITRINE_ERR_BAD_ARGUMENT -1
/** The file could not be read or written. */
#define VITRINE_ERR_IO -2
/** Rust panicked inside the boundary; treat as a cold cache and continue. */
#define VITRINE_ERR_PANIC -3
/** A record could not be represented (unknown stage, oversized code). */
#define VITRINE_ERR_BAD_RECORD -4

/* -- types --------------------------------------------------------------- */

/** Health of a cache file and the records recovered from it. */
typedef struct Report {
    int status;               /**< VITRINE_OK or VITRINE_WARN_DAMAGED, or a negative error. */
    size_t record_count;      /**< Records that parsed cleanly. */
    size_t env_count;         /**< Total environments ("shaders") across those records. */
    size_t bytes_consumed;    /**< Bytes read before parsing stopped. */
    size_t damage_offset;     /**< Offset of the first failure; 0 when clean. */
    const char* damage_text;  /**< Reserved; currently always NULL. */
    void* handle;             /**< Owned; release with vitrine_report_free(). */
} Report;

/** One environment, in the form C++ can build without touching Rust types. */
typedef struct CEntry {
    unsigned int local_memory_size;
    unsigned int texture_bound;
    unsigned int start_address;
    unsigned int cached_lowest;
    unsigned int cached_highest;
    unsigned int viewport_transform_state;
    unsigned int stage;              /**< 0=Vertex 1=TessCtrl 2=TessEval 3=Geometry 4=Fragment 5=Compute */
    const unsigned long long* code; /**< ARM64 instruction words. */
    size_t code_word_count;
    unsigned int workgroup_size[3]; /**< Compute only. */
    unsigned int shared_memory_size;/**< Compute only. */
} CEntry;

/** A pipeline to write: its 16-byte key and the environments compiled for it. */
typedef struct PendingPipeline {
    unsigned char key[16];
    const CEntry* envs;
    size_t env_count; /**< Must be 1..=64. */
} PendingPipeline;

/* -- functions ----------------------------------------------------------- */

/**
 * Read a cache file and describe its health without modifying it.
 *
 * A missing file is not an error: it reports VITRINE_OK with zero records, matching the C++
 * behaviour of returning early when the stream will not open.
 *
 * The returned Report owns a handle and MUST be released with vitrine_report_free().
 */
Report vitrine_cache_inspect(const char* path);

/**
 * Act on an inspected cache.
 *
 * @param truncate  non-zero: rewrite the file with only the records that parsed (loses the
 *                  unreadable tail and nothing else). zero: delete the file entirely.
 * @return          number of records written, or a negative error code.
 */
int vitrine_cache_repair(const char* path, Report* report, int truncate);

/**
 * Append one pipeline's environments to a cache file, creating it when absent.
 *
 * If the existing file is damaged it is repaired first, so a second header can never be written
 * into the middle of the stream.
 *
 * @return  1 on success, or a negative error code.
 */
int vitrine_cache_append(const char* path, const PendingPipeline* pipeline);

/** Release a Report and the records it holds. Safe with a NULL pointer. */
void vitrine_report_free(Report* report);

/** Version string of the linked library, as a static NUL-terminated string. Never NULL. */
const char* vitrine_version(void);

/* -- content selection ---------------------------------------------------- */

/** Title type discriminants, matching `TitleType` in `src/core/file_sys/cnmt.h`. */
#define VITRINE_TYPE_APPLICATION 0x80
#define VITRINE_TYPE_PATCH 0x81
#define VITRINE_TYPE_ADD_ON_CONTENT 0x82
#define VITRINE_TYPE_DELTA 0x83

/** Reject reason: a higher version of the same title id was chosen. */
#define VITRINE_REJECT_SUPERSEDED 0
/** Reject reason: the identical content was installed twice. */
#define VITRINE_REJECT_DUPLICATE 1

/** Warning: a patch is installed but its base application is not. */
#define VITRINE_WARN_ORPHAN_PATCH 0
/** Warning: add-on content is installed but its base application is not. */
#define VITRINE_WARN_ORPHAN_DLC 1

/** One installed CNMT. */
typedef struct CContentEntry {
    unsigned long long title_id;
    unsigned int version;
    unsigned int title_type; /**< One of the VITRINE_TYPE_* values. */
    unsigned long long source; /**< Caller's file identifier; returned untouched. */
} CContentEntry;

/** A rejected entry, why, and (for supersession) the version that won. */
typedef struct CReject {
    CContentEntry entry;
    unsigned int reason;
    unsigned int chosen_version;
} CReject;

/** A non-fatal problem worth logging. */
typedef struct CWarning {
    unsigned int kind; /**< One of the VITRINE_WARN_* values. */
    unsigned long long title_id;
    unsigned int version;
} CWarning;

/** What vitrine_content_select() produced. */
typedef struct CSelectResult {
    size_t chosen_count;   /**< True total, even when truncated. */
    size_t rejected_count; /**< True total, even when truncated. */
    size_t warning_count;  /**< True total, even when truncated. */
    unsigned int truncated;/**< Non-zero when an output array was too small. */
} CSelectResult;

/**
 * Choose one entry per title id: highest version wins, ties reported, nothing silently dropped.
 *
 * Replaces the `insert_or_assign` in RegisteredCache::ProcessFiles, where the winner was whichever
 * CNMT the directory walk happened to reach last — so two installed updates for one game booted on
 * an arbitrary version.
 *
 * Entries with an unknown `title_type` are skipped and not counted.
 *
 * @return number of chosen entries, or a negative error code.
 */
int vitrine_content_select(const CContentEntry* entries, size_t entry_count,
                           CContentEntry* chosen, size_t chosen_cap,
                           CReject* rejects, size_t reject_cap,
                           CWarning* warnings, size_t warning_cap,
                           CSelectResult* out);


/* ============================================================================================
 * The JIT ABI — a running CPU, not a file.
 *
 * The functions above describe a pipeline cache on disk. These describe execution: the caller owns
 * a guest state block, supplies a word reader that can REFUSE (an unmapped page is negative, not a
 * zero word), and gets back where execution stopped and why.
 *
 * The C++ backend for Citron's ArmInterface maps onto this directly:
 *   RunThread      -> vitrine_jit_run(max_blocks) in a loop until the exit code is not 0/1
 *   StepThread     -> vitrine_jit_run(1)
 *   InvalidateRange-> vitrine_jit_invalidate
 *   Get/SetContext -> the reg/pc/sp/flags accessors below
 * ========================================================================================== */

typedef struct VitrineJit VitrineJit;

/* Reads one 32-bit instruction word at `addr`. Return a NEGATIVE value when the address cannot be
 * read. Negative rather than a sentinel word, because every 32-bit pattern is a legal encoding. */
typedef int64_t (*vitrine_read_word)(void* ctx, uint64_t addr);

/* Exit codes, matching jit::Exit in Rust. */
#define VITRINE_EXIT_NEXT          0u
#define VITRINE_EXIT_JUMPED        1u
#define VITRINE_EXIT_SVC           2u
#define VITRINE_EXIT_UNSUPPORTED   3u
#define VITRINE_EXIT_DATA_ABORT    4u

typedef struct {
    uint64_t executed;    /* blocks retired before stopping */
    uint32_t exit_code;   /* one of VITRINE_EXIT_* */
    uint64_t pc;          /* where execution stopped; on a data abort, the faulting instruction */
    uint64_t fault_addr;  /* valid only when exit_code == VITRINE_EXIT_DATA_ABORT */
    int32_t status;       /* 0 on a completed run, negative on a refusal or a panic */
} VitrineRunResult;

/* Allocate the guest state block: the size is exposed so C++ never hard-codes the layout. */
size_t vitrine_jit_state_size(void);

/* Create a JIT. The state block is COPIED in; the caller may free theirs immediately after.
 * Returns NULL on a null state pointer or an internal panic. */
VitrineJit* vitrine_jit_new(const void* state, vitrine_read_word read, void* ctx);

/* The same, with a ZEROED guest state: use this when the caller has no state block to import.
 * Allocating vitrine_jit_state_size() bytes in C and passing them here would be uninitialised
 * memory crossing the boundary, which is undefined however well it appears to work. */
VitrineJit* vitrine_jit_new_default(vitrine_read_word read, void* ctx);

void vitrine_jit_free(VitrineJit* jit);

/* Run up to max_blocks blocks from the current pc. */
VitrineRunResult vitrine_jit_run(VitrineJit* jit, uint64_t max_blocks);

/* Execute exactly ONE instruction, translated fresh and uncached. StepThread uses this: a
 * block-granular "step" has already walked past the instruction the debugger was showing. */
VitrineRunResult vitrine_jit_step(VitrineJit* jit);

/* Drop cached translations OVERLAPPING [addr, addr+size). Returns how many were dropped.
 * Overlap, not containment: a block translated before a branch into its middle was found is stale,
 * and its start lies outside the range being invalidated. */
uint64_t vitrine_jit_invalidate(VitrineJit* jit, uint64_t addr, uint64_t size);

uint64_t vitrine_jit_cache_len(VitrineJit* jit);

/* Guest state. x0..x30 only; sp, pc and flags have their own accessors. */
uint64_t vitrine_jit_get_reg(VitrineJit* jit, uint32_t n);
void     vitrine_jit_set_reg(VitrineJit* jit, uint32_t n, uint64_t value);
uint64_t vitrine_jit_get_pc(VitrineJit* jit);
void     vitrine_jit_set_pc(VitrineJit* jit, uint64_t value);
uint64_t vitrine_jit_get_sp(VitrineJit* jit);
void     vitrine_jit_set_sp(VitrineJit* jit, uint64_t value);
uint8_t  vitrine_jit_get_flags(VitrineJit* jit);
void     vitrine_jit_set_flags(VitrineJit* jit, uint8_t value);
uint64_t vitrine_jit_fault_addr(VitrineJit* jit);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VITRINE_H */
