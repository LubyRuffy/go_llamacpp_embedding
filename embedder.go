package go_llamacpp_embedding

/*
#cgo CXXFLAGS: -std=c++17
#cgo CXXFLAGS: -O3
#cgo CXXFLAGS: -DNDEBUG
#cgo CXXFLAGS: -fPIC
#cgo CXXFLAGS: -I./llama.cpp/include
#cgo CXXFLAGS: -I./llama.cpp/common
#cgo CXXFLAGS: -I./llama.cpp/ggml/include
#cgo LDFLAGS: -L./llama.cpp/build/src -lllama
#cgo LDFLAGS: -L./llama.cpp/build/common -lcommon
#cgo LDFLAGS: -L./llama.cpp/build/ggml/src -lggml-base -lggml -lggml-cpu
#cgo LDFLAGS: -L./llama.cpp/build/ggml/src/ggml-blas -lggml-blas
#cgo darwin LDFLAGS: -L./llama.cpp/build/ggml/src/ggml-metal -lggml-metal -framework Accelerate -framework Metal -framework Foundation -framework MetalKit
#include <stdint.h>
#include <stdlib.h>
#include "llama-go.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	loadOnce sync.Once
)

const backendLogLevelError = 2

func init() {
	loadOnce.Do(func() {
		// 初始化后端
		C.load_library(C.int(backendLogLevelError)) // error 级别
	})
}

type Embedder struct {
	model   C.model_t
	context C.context_t
	dim     int
	tokens  atomic.Uint64
	mu      sync.Mutex
	closed  bool
}

func (e *Embedder) EmbedTexts(texts []string) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("embedder is closed")
	}
	switch {
	case e.model == nil || e.context == nil:
		return nil, fmt.Errorf("context is not initialized")
	case e.dim <= 0:
		return nil, fmt.Errorf("model does not support embedding")
	case len(texts) == 0:
		return [][]float32{}, nil
	}

	numRows := len(texts)
	// 为每个文本分配 C 字符串
	cStrs := make([]*C.char, len(texts))
	for i, s := range texts {
		cStrs[i] = C.CString(s)
	}
	defer func() {
		for _, p := range cStrs {
			if p != nil {
				C.free(unsafe.Pointer(p))
			}
		}
	}()

	// 在 C 侧分配 char** 数组并填充，以满足 cgo 规则（避免将含指针的 Go 切片传入 C）
	ptrBytes := C.size_t(len(texts)) * C.size_t(unsafe.Sizeof((*C.char)(nil)))
	cArray := C.malloc(ptrBytes)
	if cArray == nil {
		return nil, fmt.Errorf("failed to allocate C array for texts")
	}
	defer C.free(cArray)
	cPtrSlice := unsafe.Slice((**C.char)(cArray), len(texts))
	copy(cPtrSlice, cStrs)

	// 输出缓冲区（C.float），调用后再转换为 Go float32
	out := make([]C.float, numRows*int(e.dim))
	var rows C.uint32_t
	var toks C.uint32_t
	ret := C.embed_text_list(
		e.context,
		(**C.char)(cArray),
		C.uint32_t(len(texts)),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.uint32_t(numRows),
		(*C.uint32_t)(unsafe.Pointer(&rows)),
		(*C.uint32_t)(unsafe.Pointer(&toks)),
	)
	e.tokens.Add(uint64(toks))
	switch int(ret) {
	case 0:
		nEmb := int(e.dim)
		nRows := int(rows)
		buf := make([]float32, nRows*nEmb)
		for i := 0; i < nRows*nEmb; i++ {
			buf[i] = float32(out[i])
		}
		res := make([][]float32, nRows)
		for i := 0; i < nRows; i++ {
			start := i * nEmb
			res[i] = buf[start : start+nEmb]
		}
		return res, nil
	case 1:
		return nil, fmt.Errorf("a prompt exceeds batch size")
	case 4:
		return nil, fmt.Errorf("pooling type NONE not supported for batch API")
	case 5:
		return nil, fmt.Errorf("insufficient output rows buffer: need >= number of prompts")
	default:
		return nil, fmt.Errorf("failed to embed texts (code=%d)", int(ret))
	}
}

func (e *Embedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	C.free_context(e.context)
	C.free_model(e.model)
	e.context = nil
	e.model = nil
	e.closed = true
}

func (e *Embedder) Tokens() uint64 {
	return e.tokens.Load()
}

func NewEmbedder(modelPath string, gpuLayers int) (*Embedder, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	model := C.load_model(cPath, C.int(gpuLayers))
	if model == nil {
		return nil, fmt.Errorf("加载模型失败")
	}
	context := C.load_context(model, 0, true)
	if context == nil {
		C.free_model(model)
		return nil, fmt.Errorf("创建上下文失败")
	}
	dim := C.embed_size(model)
	if dim <= 0 {
		C.free_context(context)
		C.free_model(model)
		return nil, fmt.Errorf("当前模型不支持 embedding")
	}
	return &Embedder{model: model, context: context, dim: int(dim)}, nil
}
