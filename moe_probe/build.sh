#!/usr/bin/env bash
# Build llama.cpp (shared libs) and the moe_probe binary that links against it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HERE/../llama.cpp}"
[ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF ${CMAKE_EXTRA:-}
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama

LIB="$LLAMA_DIR/build/bin"
mkdir -p "$HERE/build"
g++ -O2 -std=c++17 "$HERE/moe_probe.cpp" -o "$HERE/build/moe_probe" \
    -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" -I"$LLAMA_DIR/vendor" \
    -L"$LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"
echo "built $HERE/build/moe_probe"
