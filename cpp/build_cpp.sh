#!/usr/bin/env bash
# Build the C++ integration check against the compiled Rust library.
#
# This is the same link line a fork of Citron Neo would use:
#   - build the crate          -> libvitrine.so
#   - add include/ to the path -> vitrine.h
#   - link -lvitrine
#
# Usage: ./cpp/build_cpp.sh [--static]
#   --static links libvitrine.a instead, for a fully static binary.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"

if [ "$MODE" = "--static" ]; then
    cargo build --release
    g++ -O2 -std=c++20 -Iinclude cpp/cpp_check.cpp \
        target/release/libvitrine.a -o cpp/cpp_check-static
    echo "built cpp/cpp_check-static (static)"
    exit 0
fi

cargo build --release
# rpath points at the build tree so the check runs without installing anything.
g++ -O2 -std=c++20 -Iinclude cpp/cpp_check.cpp -o cpp/cpp_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"
echo "built cpp/cpp_check"
