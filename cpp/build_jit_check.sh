#!/usr/bin/env bash
# Build the JIT half of the C++ integration check against the Rust library.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release
g++ -O2 -std=c++20 -Iinclude cpp/jit_check.cpp -o cpp/jit_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"
echo "built cpp/jit_check"
