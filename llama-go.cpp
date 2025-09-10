#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include <limits.h>

typedef struct llama_model* model_t;
typedef struct llama_context* context_t;
int32_t embd_normalize = 2; // normalization (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)


static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), true);

    // run model
    // LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

extern "C" {

    // Load the library and initialize the backend
    LLAMA_API void load_library(ggml_log_level desired) {
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

        // Set the log level
        llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, NULL);
    }

    // Load the model from the file
    LLAMA_API model_t load_model(const char * path_model, const int32_t n_gpu_layers) {
        struct llama_model_params params = llama_model_default_params();
        // emulate llama.cpp examples: -1 means offload as many as possible
        params.n_gpu_layers = (n_gpu_layers < 0) ? INT32_MAX : n_gpu_layers;

        return llama_model_load_from_file(path_model, params); // Updated function
    }

    // Free the model and all resources
    LLAMA_API void free_model(model_t model) {
        llama_model_free(model); // Updated function
    }

    // Create a context with the model and the specified context size
    LLAMA_API context_t load_context(model_t model, const uint32_t ctx_size, const bool embeddings) {
        struct llama_context_params params = llama_context_default_params();
        // Default to example's typical context size when not specified
        params.n_ctx = ctx_size == 0 ? 4096 : ctx_size;
        params.embeddings = embeddings; // Corrected field name
        // utilize the full context only when needed
        if (params.n_batch < params.n_ctx) {
            LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
            params.n_batch = params.n_ctx;
        }
        // For non-causal models, batch size must equal ubatch size
        params.n_ubatch = params.n_batch;
        // Fallback to unified KV cache to support any number of prompts when parallel is not specified
        params.kv_unified = true;

        return llama_init_from_model(model, params); // Updated function
    }

    // Free the context and all resources
    LLAMA_API void free_context(context_t ctx) {
        llama_free(ctx);
    }

    // Get the embedding size, return -1 if model doesn't support embeddings
    LLAMA_API int32_t embed_size(model_t model) {
        if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
            return -1; // Embeddings not supported for encoder-decoder models
        }
        return llama_model_n_embd(model);
    }

    // Embed the text and return the embeddings
    LLAMA_API int embed_text(context_t ctx, const char* text, float* out_embeddings, uint32_t* out_tokens) {
        const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
        model_t model = (model_t)llama_get_model(ctx);
        const uint64_t n_batch = llama_n_batch(ctx);

        // Tokenize the input text
        auto inp = common_tokenize(ctx, text, true, true);
        *out_tokens = inp.size();
        if (inp.size() > n_batch) {
            printf("Number of tokens exceeds batch size, increase batch size\n");
            return 1; // Number of tokens exceeds batch size
        }

        // Check if the last token is SEP/EOS; warn if not (align with example behavior)
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (inp.empty() || (inp.back() != llama_vocab_sep(vocab) && inp.back() != llama_vocab_eos(vocab))) {
            LOG_WRN("%s: last token in the prompt is not SEP or EOS\n", __func__);
            LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
        }

        // Initialize batch
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
        batch_add_seq(batch, inp, 0);

        // Decode batch and store embeddings in out_embeddings
        const int n_embd = llama_model_n_embd(model);
        batch_decode(ctx, batch, out_embeddings, 1, n_embd, embd_normalize);

        // Clean up
        llama_batch_free(batch);
        return 0;
    }

    // Embed multiple prompts via string array (no separator parsing).
    // - texts: array of pointers to null-terminated strings
    // - num_texts: number of prompts
    // - out_embeddings: float array with capacity >= max_rows * n_embd
    // - max_rows: maximum number of prompt embeddings to write
    // - out_rows: actual number of prompt embeddings written
    // - out_tokens: total number of tokens processed
    // Returns:
    //  0 success
    //  1 some input exceeds batch size
    //  3 generic failure
    //  4 unsupported when pooling type is NONE (token-level embeddings produce ragged output)
    //  5 insufficient output rows buffer (max_rows < number of prompts)
    LLAMA_API int embed_text_list(context_t ctx, const char** texts, uint32_t num_texts, float* out_embeddings, uint32_t max_rows, uint32_t* out_rows, uint32_t* out_tokens) {
        if (ctx == nullptr || texts == nullptr || out_embeddings == nullptr || out_rows == nullptr || out_tokens == nullptr) {
            return 3;
        }

        const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            return 4;
        }

        model_t model = (model_t)llama_get_model(ctx);
        const uint64_t n_batch = llama_n_batch(ctx);
        const int n_embd = llama_model_n_embd(model);

        const int n_prompts = (int) num_texts;
        if (max_rows < (uint32_t)n_prompts) {
            return 5;
        }

        std::vector<std::vector<int32_t>> inputs;
        inputs.reserve(n_prompts);

        uint64_t total_tokens = 0;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        for (int i = 0; i < n_prompts; ++i) {
            const char* prompt = texts[i];
            std::vector<int32_t> inp = common_tokenize(ctx, prompt, true, true);
            total_tokens += inp.size();

            if (inp.size() > n_batch) {
                return 1;
            }

            if (inp.empty() || (inp.back() != llama_vocab_sep(vocab) && inp.back() != llama_vocab_eos(vocab))) {
                LOG_WRN("%s: last token in the prompt is not SEP or EOS\n", __func__);
                LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
            }

            inputs.push_back(std::move(inp));
        }

        struct llama_batch batch = llama_batch_init(n_batch, 0, n_prompts);

        int e = 0;
        int s = 0;
        for (int k = 0; k < n_prompts; k++) {
            auto & inp = inputs[k];
            const uint64_t n_toks = inp.size();
            if (batch.n_tokens + n_toks > n_batch) {
                float * out = out_embeddings + e * n_embd;
                batch_decode(ctx, batch, out, s, n_embd, embd_normalize);
                e += s;
                s = 0;
                common_batch_clear(batch);
            }
            batch_add_seq(batch, inp, s);
            s += 1;
        }

        float * out = out_embeddings + e * n_embd;
        batch_decode(ctx, batch, out, s, n_embd, embd_normalize);
        e += s;

        *out_rows = (uint32_t)e;
        *out_tokens = (uint32_t)total_tokens;

        llama_batch_free(batch);
        return 0;
    }
}
