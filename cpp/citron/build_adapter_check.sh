#!/usr/bin/env bash
# Build the ArmInterface adapter check. Include paths: the shim, this directory, and vitrine.h.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release
g++ -O2 -std=c++20 -Wall -Wextra \
    -Iinclude -Icpp/citron -Icpp/citron/shim \
    cpp/citron/adapter_check.cpp cpp/citron/arm_vitrine_64.cpp -o cpp/citron/adapter_check \
    -Ltarget/release -lvitrine -Wl,-rpath,"$PWD/target/release"
echo "built cpp/citron/adapter_check"
