package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/LubyRuffy/go_llamacpp_embedding"
)

func main() {
	modelPath := flag.String("model", "all-MiniLM-L6-v2-Q8_0.gguf", "GGUF 模型路径")
	prompt := flag.String("prompt", "hello world", "要生成 embedding 的文本")
	gpuLayers := flag.Int("gpu-layers", -1, "启用的 GPU 层数，0=CPU，99=尽可能多")
	flag.Parse()

	embedder, err := go_llamacpp_embedding.NewEmbedder(*modelPath, *gpuLayers)
	if err != nil {
		log.Fatal(err)
	}
	defer embedder.Close()

	// 单文本 embedding
	out, err := embedder.EmbedTexts([]string{*prompt})
	if err != nil {
		panic(err)
	}
	fmt.Printf("embedding=%v\n", out)
}
