# go_llamacpp_embedding

golang的embedding库，不依赖外部工具直接使用gguf模型。
内嵌到golang程序里面的调用llama.cpp实现embedding的库。

## 编译

```bash
./build.sh
```

会自动下载llama.cpp和all-MiniLM-L6-v2-Q8_0.gguf，并编译。

## 使用

```go
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
```
