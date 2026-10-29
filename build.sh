#!/usr/bin/env bash
# One command that runs everything CI runs, in the same order.
#
#   ./build.sh
#
# A fork should be able to clone, run this, and know whether it is healthy. If a step here can pass
# while the same step fails on GitHub, this script is wrong — keep them identical.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n== %s\n' "$1"; }

say "1/5 format"
cargo fmt --all -- --check

say "2/5 clippy (warnings are errors)"
cargo clippy --release --all-targets -- -D warnings

say "3/5 tests"
cargo test --release

say "4/5 C++ boundary (both programs built and RUN)"
cargo build --release
g++ -O2 -std=c++20 -Iinclude cpp/jit_check.cpp -o cpp/jit_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"
./cpp/jit_check
g++ -O2 -std=c++20 -Wall -Wextra -Werror \
    -Iinclude -Icpp/citron -Icpp/citron/shim \
    cpp/citron/adapter_check.cpp cpp/citron/arm_vitrine_64.cpp -o cpp/citron/adapter_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"
./cpp/citron/adapter_check
g++ -O2 -std=c++20 -Iinclude cpp/cpp_check.cpp -o cpp/cpp_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"

say "5/5 cmake package config"
PREFIX="$(mktemp -d)/vitrine-prefix"
mkdir -p "$PREFIX/lib/cmake/vitrine" "$PREFIX/include"
cp target/release/libvitrine.a "$PREFIX/lib/"
cp include/vitrine.h "$PREFIX/include/"
cp cmake/vitrineConfig.cmake cmake/vitrineConfigVersion.cmake "$PREFIX/lib/cmake/vitrine/"
WORK="$(mktemp -d)/pkgtest"
mkdir -p "$WORK"
printf '%s\n' '#include <stdio.h>' '#include "vitrine.h"' 'int main(void){' \
    '  printf("vitrine %s, state block %zu bytes\n", vitrine_version(), vitrine_jit_state_size());' \
    '  return vitrine_jit_state_size() == 0 ? 1 : 0; }' > "$WORK/main.c"
printf '%s\n' 'cmake_minimum_required(VERSION 3.20)' 'project(pkgtest C)' \
    'find_package(vitrine 0.1 CONFIG REQUIRED)' 'add_executable(pkgtest main.c)' \
    'target_link_libraries(pkgtest PRIVATE vitrine::vitrine)' > "$WORK/CMakeLists.txt"
cmake -S "$WORK" -B "$WORK/build" -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$WORK/build" >/dev/null
"$WORK/build/pkgtest"

printf '\nALL STEPS PASSED\n'
