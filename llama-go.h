// Minimal C header for exported functions
#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_model* model_t;
typedef struct llama_context* context_t;

void load_library(int desired);
model_t load_model(const char * path_model, const int32_t n_gpu_layers);
void free_model(model_t model);
context_t load_context(model_t model, const uint32_t ctx_size, const bool embeddings);
void free_context(context_t ctx);
int32_t embed_size(model_t model);
int embed_text(context_t ctx, const char* text, float* out_embeddings, uint32_t* out_tokens);
int embed_text_list(context_t ctx, const char** texts, uint32_t num_texts, float* out_embeddings, uint32_t max_rows, uint32_t* out_rows, uint32_t* out_tokens);

#ifdef __cplusplus
}
#endif

