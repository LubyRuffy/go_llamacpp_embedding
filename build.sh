#!/usr/bin/env bash
set -euo pipefail

# 如果llama.cpp目录不存在，则clone
if [ ! -d "./llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp
  # 静态库
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release -j
  cd ..
fi

# 如果all-MiniLM-L6-v2-Q8_0.gguf不存在，则下载
if [ ! -e "all-MiniLM-L6-v2-Q8_0.gguf" ]; then
  wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf
fi

# 源码与构建目录（可通过环境覆盖）
LLAMACPP_SRDIR="./llama.cpp"
LIBLLAMA_FILEPATH=`find ${LLAMACPP_SRDIR} -name 'libllama.a'`
if [ -z "${LIBLLAMA_FILEPATH}" ]; then
  echo "libllama not found"
  exit 1
fi
# 多个libllama文件，取第一个
LIBLLAMA_FILEPATH=$(echo ${LIBLLAMA_FILEPATH} | head -n 1)
LLAMACPP_LIB_DIR=$(dirname ${LIBLLAMA_FILEPATH})

# export CGO_ENABLED=1
export CGO_CXXFLAGS="-std=c++17 -O3 -DNDEBUG -fPIC \
  -I${LLAMACPP_SRDIR}/include \
  -I${LLAMACPP_SRDIR}/common \
  -I${LLAMACPP_SRDIR}/ggml/include"

# 基础静态库
export CGO_LDFLAGS="-L${LLAMACPP_LIB_DIR} -lllama \
    -L${LLAMACPP_LIB_DIR}/../common -lcommon\
    -L${LLAMACPP_LIB_DIR}/../ggml/src -lggml-base -lggml -lggml-cpu\
    -L${LLAMACPP_LIB_DIR}/../ggml/src/ggml-blas -lggml-blas\
    -L${LLAMACPP_LIB_DIR}/../ggml/src/ggml-metal -lggml-metal -framework Accelerate -framework Metal -framework Foundation -framework MetalKit"

echo "==> LLAMACPP_SRDIR=${LLAMACPP_SRDIR}"
echo "==> LLAMACPP_LIB_DIR=${LLAMACPP_LIB_DIR}"
echo "==> CGO_CXXFLAGS=${CGO_CXXFLAGS}"
echo "==> CGO_LDFLAGS=${CGO_LDFLAGS}"

go build -buildvcs=false ./examples/embedder