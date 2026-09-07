/**
 * @file argus_text.cc
 * @brief Model loading, text generation, vocabulary tokenization, KV-cache management,
 *        and speech synthesis engine.
 *
 * Implements the unmanaged execution pathways mapped to the public text handling
 * boundaries of libargus.h, maintaining safe C ABI structures for Project Panama.
 */

#include "libargus.h"
#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <mutex>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "argus_internal.h"

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * (float)M_PI * i) / (length + offset)));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2.0f * (float)M_PI * k / N;
    *real = cosf(angle);
    *imag = sinf(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real, float * scratch_real_inp, float * scratch_imag_inp, float * scratch_real_out, float * scratch_imag_out) {
    int N = n / 2 + 1;

    for (int i = 0; i < N; ++i) {
        scratch_real_inp[i] = inp_cplx[2 * i];
        scratch_imag_inp[i] = inp_cplx[2 * i + 1];
    }

    for (int k = 0; k < n; ++k) {
        scratch_real_out[k] = 0.0f;
        scratch_imag_out[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            scratch_real_out[k] += scratch_real_inp[m] * twiddle_real - scratch_imag_inp[m] * twiddle_imag;
            scratch_imag_out[k] += scratch_real_inp[m] * twiddle_imag + scratch_imag_inp[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = scratch_real_out[i] / N;
    }
}

static void fold(const float * data, int64_t data_size, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, float * output) {
    std::fill(output, output + n_out, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < n_out; ++w_col) {
        int64_t start = w_col * n_hop - n_pad;
        int64_t end   = start + n_win;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < n_out && col_idx < data_size) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }
}

static std::string process_text_for_outetts(const std::string & text) {
    std::string processed;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            processed += std::tolower(static_cast<unsigned char>(c));
        } else {
            processed += ' ';
        }
    }
    std::string trimmed;
    bool in_space = false;
    for (char c : processed) {
        if (c == ' ') {
            if (!in_space) {
                trimmed += ' ';
                in_space = true;
            }
        } else {
            trimmed += c;
            in_space = false;
        }
    }
    if (!trimmed.empty() && trimmed.front() == ' ') trimmed.erase(0, 1);
    if (!trimmed.empty() && trimmed.back() == ' ') trimmed.pop_back();

    std::string result;
    for (char c : trimmed) {
        if (c == ' ') {
            result += "<|text_sep|>";
        } else {
            result += c;
        }
    }
    return result;
}

static int32_t tokenize_to_vector(const struct llama_vocab * vocab, const std::string & text, std::vector<llama_token> & output, bool add_bos) {
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_bos, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }
    std::vector<llama_token> tokens(n_tokens);
    int32_t result = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), n_tokens, add_bos, true);
    if (result < 0) {
        return result;
    }
    output.insert(output.end(), tokens.begin(), tokens.begin() + result);
    return result;
}

extern "C" {

// =========================================================================
// Model Lifecycle
// =========================================================================

bool argus_model_retain(argus_model_t * model) {
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "model pointer is NULL");
        return false;
    }
    model->refs.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void argus_model_release(argus_model_t * model) {
    if (!model) {
        return;
    }
    if (model->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (model->model) {
            llama_model_free(model->model);
            model->model = nullptr;
        }
        delete model;
        argus_backend_resource_dec();
    }
}

void argus_model_free(argus_model_t * model) {
    argus_model_release(model);
}

argus_model_t * argus_model_load(const argus_model_params_t * params) {
    try {
        clear_last_error();
        if (!params || !params->model_path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "model params or model path is NULL");
            return nullptr;
        }

        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = params->gpu_layers;
        if (params->use_mlock) {
            mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
        }

        struct llama_model * model = llama_model_load_from_file(params->model_path, mparams);
        if (!model) {
            set_last_error(ARGUS_ERROR_MODEL_LOAD, "failed to load llama model from file");
            return nullptr;
        }

        argus_model_t * argus_model_ptr = nullptr;
        try {
            argus_model_ptr = new argus_model();
            argus_model_ptr->refs = 1;
            argus_model_ptr->model = model;
            argus_model_ptr->vocab = llama_model_get_vocab(model);
            argus_backend_resource_inc();
            return argus_model_ptr;
        } catch (...) {
            llama_model_free(model);
            throw;
        }
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_MODEL_LOAD, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during model load");
    }
    return nullptr;
}

// =========================================================================
// Context Lifecycle
// =========================================================================

argus_context_t * argus_context_init(argus_model_t * model, const argus_context_params_t * params) {
    try {
        clear_last_error();
        if (!model || !model->model || !params) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid model or context parameters");
            return nullptr;
        }

        // Retain model reference before proceeding
        if (!argus_model_retain(model)) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "failed to retain primary model");
            return nullptr;
        }

        argus_model_t * draft_model = const_cast<argus_model_t *>(params->draft_model);
        if (draft_model) {
            if (!argus_model_retain(draft_model)) {
                argus_model_release(model);
                set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "failed to retain draft model");
                return nullptr;
            }
        }

        struct llama_context_params cparams = llama_context_default_params();
        int32_t limit_batch = (params->context_length < 2048) ? params->context_length : 2048;

        int32_t seq_max = params->n_seq_max;
        if (seq_max <= 0) {
            bool requires_multi_seq = (params->draft_model != nullptr) || 
                                      (params->spec_draft_n_max > 0) || 
                                      params->enable_draft_mtp;
            seq_max = requires_multi_seq ? 4 : 1;
        }

        bool kv_unified_active = params->kv_unified || (params->n_seq_max == 0);

        // Bound the batch size by the sequence slot size to prevent KV cache slot allocation crashes.
        // Account for GGML padding of slot sizes to multiples of 256 cells.
        int32_t n_ctx_seq = kv_unified_active ? params->context_length : (params->context_length / seq_max);
        n_ctx_seq = GGML_PAD(n_ctx_seq, 256);
        if (n_ctx_seq < limit_batch) {
            limit_batch = n_ctx_seq;
        }

        // Bound the batch size by the base model's sliding window size (SWA) if present
        if (model->model) {
            int32_t base_swa = llama_model_n_swa(model->model);
            if (base_swa > 0 && base_swa < limit_batch) {
                limit_batch = base_swa;
            }
        }

        // Bounding by the draft model's sliding window size (SWA) if present to prevent draft context crash
        if (draft_model && draft_model->model) {
            int32_t draft_swa = llama_model_n_swa(draft_model->model);
            if (draft_swa > 0 && draft_swa < limit_batch) {
                limit_batch = draft_swa;
            }
        }

        // Ensure we have a valid batch size of at least 1
        if (limit_batch < 1) {
            limit_batch = 1;
        }

        cparams.n_ctx           = params->context_length;
        cparams.n_batch         = limit_batch;

        // Calculate physical micro-batch size (n_ubatch)
        int32_t req_ubatch = params->u_batch;
        if (req_ubatch <= 0) {
            bool is_encoder_or_embed = params->embeddings || (model->model && llama_model_has_encoder(model->model));
            req_ubatch = is_encoder_or_embed ? limit_batch : ((limit_batch < 512) ? limit_batch : 512);
        }
        cparams.n_ubatch        = std::clamp(req_ubatch, 1, limit_batch);
        cparams.n_seq_max       = seq_max;
        cparams.kv_unified      = kv_unified_active;
        cparams.n_threads       = params->cpu_threads;
        cparams.n_threads_batch = params->cpu_threads;

        // Multi-Token Prediction (MTP) context type configuration
        cparams.ctx_type = params->enable_draft_mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;

        // KV cache quantization support
        cparams.type_k = (enum ggml_type)params->type_k;
        cparams.type_v = (enum ggml_type)params->type_v;

        // Embeddings support
        bool is_encoder_model = model->model && llama_model_has_encoder(model->model);
        cparams.embeddings = params->embeddings || is_encoder_model;

        struct llama_context * ctx = llama_init_from_model(model->model, cparams);
        if (!ctx) {
            if (draft_model) argus_model_release(draft_model);
            argus_model_release(model);
            set_last_error(ARGUS_ERROR_BACKEND, "failed to initialize llama context");
            return nullptr;
        }

        if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            llama_set_embeddings(ctx, true);
        }

        struct llama_context * draft_ctx = nullptr;
        if (draft_model) {
            if (!draft_model->model) {
                llama_free(ctx);
                argus_model_release(draft_model);
                argus_model_release(model);
                set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "draft model has null llama model");
                return nullptr;
            }
            struct llama_context_params dparams = llama_context_default_params();
            dparams.n_ctx           = params->context_length;
            dparams.n_batch         = cparams.n_batch;
            dparams.n_ubatch        = cparams.n_ubatch;
            dparams.n_seq_max       = cparams.n_seq_max;
            dparams.kv_unified      = cparams.kv_unified;
            dparams.n_threads       = params->cpu_threads;
            dparams.n_threads_batch = params->cpu_threads;
            dparams.type_k          = cparams.type_k;
            dparams.type_v          = cparams.type_v;

            draft_ctx = llama_init_from_model(draft_model->model, dparams);
            if (!draft_ctx) {
                llama_free(ctx);
                argus_model_release(draft_model);
                argus_model_release(model);
                set_last_error(ARGUS_ERROR_BACKEND, "failed to initialize draft context");
                return nullptr;
            }
        }

        argus_context_t * argus_ctx = nullptr;
        try {
            argus_ctx = new argus_context();
            argus_ctx->ctx                  = ctx;
            argus_ctx->draft_ctx            = draft_ctx;
            argus_ctx->model_ref            = model;
            argus_ctx->draft_model_ref      = draft_model;
            argus_ctx->spec_draft_n_max     = params->spec_draft_n_max;
            argus_ctx->enable_draft_mtp     = params->enable_draft_mtp;
            argus_ctx->vocoder_ctx          = nullptr;
            argus_ctx->vocoder_model_ref    = nullptr;
            argus_ctx->last_decoded_seq_id  = -1;

            argus_ctx->seq_samplers.resize(seq_max);
            for (int32_t i = 0; i < seq_max; ++i) {
                argus_ctx->seq_samplers[i].chain = nullptr;
                argus_ctx->seq_samplers[i].has_cached_chain = false;
                argus_ctx->seq_samplers[i].has_logits = false;
                argus_ctx->seq_samplers[i].last_logits_pos = -1;
                if (params->context_length > 0) {
                    argus_ctx->seq_samplers[i].history.reserve((size_t)params->context_length);
                }
            }

            return argus_ctx;
        } catch (...) {
            if (draft_ctx) llama_free(draft_ctx);
            if (ctx) llama_free(ctx);
            if (draft_model) argus_model_release(draft_model);
            argus_model_release(model);
            if (argus_ctx) delete argus_ctx;
            throw;
        }
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during context initialization");
    }
    return nullptr;
}

void argus_context_free(argus_context_t * ctx) {
    if (!ctx) {
        return;
    }
    try {
        clear_last_error();
        for (auto & slot : ctx->seq_samplers) {
            if (slot.chain) {
                llama_sampler_free(slot.chain);
                slot.chain = nullptr;
            }
        }
        ctx->seq_samplers.clear();
        if (ctx->ctx) {
            llama_free(ctx->ctx);
            ctx->ctx = nullptr;
        }
        if (ctx->draft_ctx) {
            llama_free(ctx->draft_ctx);
            ctx->draft_ctx = nullptr;
        }
        if (ctx->vocoder_ctx) {
            llama_free(ctx->vocoder_ctx);
            ctx->vocoder_ctx = nullptr;
        }
        if (ctx->vocoder_model_ref) {
            argus_model_release(const_cast<argus_model_t *>(ctx->vocoder_model_ref));
            ctx->vocoder_model_ref = nullptr;
        }
        if (ctx->draft_model_ref) {
            argus_model_release(ctx->draft_model_ref);
            ctx->draft_model_ref = nullptr;
        }
        if (ctx->model_ref) {
            argus_model_release(ctx->model_ref);
            ctx->model_ref = nullptr;
        }
        delete ctx;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during context free");
    }
}


void argus_set_n_threads(argus_context_t * ctx, int32_t n_threads, int32_t n_threads_batch) {
    argus_guard_void("argus_set_n_threads", [&]() {
        if (!ctx) {
            return;
        }
        std::lock_guard<std::mutex> lock(ctx->mtx);
        if (ctx->ctx) {
            llama_set_n_threads(ctx->ctx, n_threads, n_threads_batch);
        }
        if (ctx->draft_ctx) {
            llama_set_n_threads(ctx->draft_ctx, n_threads, n_threads_batch);
        }
        if (ctx->vocoder_ctx) {
            llama_set_n_threads(ctx->vocoder_ctx, n_threads, n_threads_batch);
        }
    });
}

int32_t argus_get_n_threads(argus_context_t * ctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_get_n_threads", [&]() -> int32_t {
        if (!ctx) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(ctx->mtx);
        if (!ctx->ctx) {
            return -1;
        }
        return llama_n_threads(ctx->ctx);
    });
}

int32_t argus_get_n_threads_batch(argus_context_t * ctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_get_n_threads_batch", [&]() -> int32_t {
        if (!ctx) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(ctx->mtx);
        if (!ctx->ctx) {
            return -1;
        }
        return llama_n_threads_batch(ctx->ctx);
    });
}

bool argus_context_has_draft(const argus_context_t * ctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_context_has_draft", [&]() -> bool {
        if (!ctx) {
            return false;
        }
        std::lock_guard<std::mutex> lock(const_cast<argus_context_t *>(ctx)->mtx);
        return ctx->draft_ctx != nullptr;
    });
}

// =========================================================================
// Tokenizer (Lock-Free, Read-Only Model Vocabulary Operations)
// =========================================================================

int32_t argus_tokenize_n(const argus_model_t * model, const char * text, size_t text_len, int32_t * out_tokens, int32_t max_tokens, bool add_bos) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_tokenize_n", [&]() -> int32_t {
        if (!model || !model->vocab || !text || !out_tokens || max_tokens <= 0 || text_len > (size_t)INT32_MAX) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid arguments to argus_tokenize_n");
            return -1;
        }
        return llama_tokenize(model->vocab, text, (int32_t)text_len, out_tokens, max_tokens, add_bos, true);
    });
}

int32_t argus_tokenize(const argus_model_t * model, const char * text, int32_t * out_tokens, int32_t max_tokens, bool add_bos) {
    if (!text) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "text pointer is NULL");
        return -1;
    }
    return argus_tokenize_n(model, text, std::strlen(text), out_tokens, max_tokens, add_bos);
}

int32_t argus_token_to_piece(const argus_model_t * model, int32_t token, char * out_buf, int32_t buf_size) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_token_to_piece", [&]() -> int32_t {
        if (!model || !model->vocab || !out_buf || buf_size <= 0 || token < 0) {
            return -1;
        }
        return llama_token_to_piece(model->vocab, token, out_buf, buf_size, 0, true);
    });
}

int32_t argus_vocab_bos(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_vocab_bos", [&]() -> int32_t {
        return model && model->vocab ? (int32_t)llama_vocab_bos(model->vocab) : -1;
    });
}

int32_t argus_vocab_eos(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_vocab_eos", [&]() -> int32_t {
        return model && model->vocab ? (int32_t)llama_vocab_eos(model->vocab) : -1;
    });
}

int32_t argus_vocab_eot(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_vocab_eot", [&]() -> int32_t {
        return model && model->vocab ? (int32_t)llama_vocab_eot(model->vocab) : -1;
    });
}

int32_t argus_vocab_pad(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_vocab_pad", [&]() -> int32_t {
        return model && model->vocab ? (int32_t)llama_vocab_pad(model->vocab) : -1;
    });
}

int32_t argus_vocab_n_tokens(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_vocab_n_tokens", [&]() -> int32_t {
        return model && model->vocab ? (int32_t)llama_vocab_n_tokens(model->vocab) : -1;
    });
}

bool argus_vocab_is_eog(const argus_model_t * model, int32_t token) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_vocab_is_eog", [&]() -> bool {
        return (model && model->vocab && token >= 0) ? llama_vocab_is_eog(model->vocab, (llama_token)token) : false;
    });
}

int32_t argus_model_meta_val_str_n(const argus_model_t * model, const char * key, size_t key_len, char * buf, int32_t buf_size) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_meta_val_str_n", [&]() -> int32_t {
        if (!model || !model->model || !key || !buf || buf_size <= 0 || key_len == 0 || key_len > (size_t)INT32_MAX) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid arguments to argus_model_meta_val_str_n");
            return -1;
        }
        std::string key_str(key, key_len);
        return llama_model_meta_val_str(model->model, key_str.c_str(), buf, (size_t)buf_size);
    });
}

int32_t argus_model_meta_val_str(const argus_model_t * model, const char * key, char * buf, int32_t buf_size) {
    if (!key) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "key is NULL");
        return -1;
    }
    return argus_model_meta_val_str_n(model, key, std::strlen(key), buf, buf_size);
}

int32_t argus_model_meta_count(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_meta_count", [&]() -> int32_t {
        if (!model || !model->model) {
            return -1;
        }
        return llama_model_meta_count(model->model);
    });
}

int32_t argus_model_meta_key_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_meta_key_by_index", [&]() -> int32_t {
        if (!model || !model->model || !buf || buf_size <= 0 || index < 0) {
            return -1;
        }
        return llama_model_meta_key_by_index(model->model, index, buf, (size_t)buf_size);
    });
}

int32_t argus_model_meta_val_str_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_meta_val_str_by_index", [&]() -> int32_t {
        if (!model || !model->model || !buf || buf_size <= 0 || index < 0) {
            return -1;
        }
        return llama_model_meta_val_str_by_index(model->model, index, buf, (size_t)buf_size);
    });
}

int32_t argus_model_n_embd(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_embd", [&]() -> int32_t {
        return model && model->model ? llama_model_n_embd(model->model) : -1;
    });
}

int32_t argus_model_n_ctx_train(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_ctx_train", [&]() -> int32_t {
        return model && model->model ? llama_model_n_ctx_train(model->model) : -1;
    });
}

int32_t argus_model_n_layer(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_layer", [&]() -> int32_t {
        return model && model->model ? llama_model_n_layer(model->model) : -1;
    });
}

int32_t argus_model_n_head(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_head", [&]() -> int32_t {
        return model && model->model ? llama_model_n_head(model->model) : -1;
    });
}

int32_t argus_model_n_head_kv(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_head_kv", [&]() -> int32_t {
        return model && model->model ? llama_model_n_head_kv(model->model) : -1;
    });
}

uint64_t argus_model_n_params(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, 0ULL, "argus_model_n_params", [&]() -> uint64_t {
        return model && model->model ? llama_model_n_params(model->model) : 0ULL;
    });
}

bool argus_model_has_encoder(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_model_has_encoder", [&]() -> bool {
        return model && model->model ? llama_model_has_encoder(model->model) : false;
    });
}

int32_t argus_model_n_pos_per_embd(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_n_pos_per_embd", [&]() -> int32_t {
        if (!model || !model->model) {
            return -1;
        }
        enum llama_rope_type rtype = llama_model_rope_type(model->model);
        return (rtype == LLAMA_ROPE_TYPE_MROPE || rtype == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    });
}

bool argus_model_is_mrope(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_model_is_mrope", [&]() -> bool {
        if (!model || !model->model) {
            return false;
        }
        enum llama_rope_type rtype = llama_model_rope_type(model->model);
        return (rtype == LLAMA_ROPE_TYPE_MROPE || rtype == LLAMA_ROPE_TYPE_IMROPE);
    });
}

uint64_t argus_model_size(const argus_model_t * model) {
    return argus_guard(ARGUS_ERROR_INTERNAL, 0ULL, "argus_model_size", [&]() -> uint64_t {
        return model && model->model ? llama_model_size(model->model) : 0ULL;
    });
}

int32_t argus_model_desc(const argus_model_t * model, char * buf, int32_t buf_size) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_model_desc", [&]() -> int32_t {
        if (!model || !model->model || !buf || buf_size <= 0) {
            return -1;
        }
        return llama_model_desc(model->model, buf, (size_t)buf_size);
    });
}

static enum ggml_type to_ggml_type(int32_t type) {
    if (type == ARGUS_KV_TYPE_F16) {
        return GGML_TYPE_F16;
    }
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return GGML_TYPE_F16;
    }
    return static_cast<enum ggml_type>(type);
}

size_t argus_quant_type_size(int32_t type) {
    return argus_guard(ARGUS_ERROR_INTERNAL, 0ULL, "argus_quant_type_size", [&]() -> size_t {
        if (type < 0 || (type >= GGML_TYPE_COUNT && type != ARGUS_KV_TYPE_F16)) {
            return 0;
        }
        enum ggml_type gtype = to_ggml_type(type);
        return ggml_type_size(gtype);
    });
}

int32_t argus_quant_block_size(int32_t type) {
    return argus_guard(ARGUS_ERROR_INTERNAL, 0, "argus_quant_block_size", [&]() -> int32_t {
        if (type < 0 || (type >= GGML_TYPE_COUNT && type != ARGUS_KV_TYPE_F16)) {
            return 0;
        }
        enum ggml_type gtype = to_ggml_type(type);
        return (int32_t)ggml_blck_size(gtype);
    });
}

int64_t argus_model_kv_bytes_per_token(const argus_model_t * model, int32_t type_k, int32_t type_v) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1LL, "argus_model_kv_bytes_per_token", [&]() -> int64_t {
        if (!model || !model->model) {
            return -1;
        }

        int32_t n_layer = argus_model_n_layer(model);
        int32_t n_head = argus_model_n_head(model);
        int32_t n_head_kv = argus_model_n_head_kv(model);
        int32_t n_embd = argus_model_n_embd(model);

        if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_embd <= 0) {
            return -1;
        }

        enum ggml_type gk = to_ggml_type(type_k);
        enum ggml_type gv = to_ggml_type(type_v);

        int32_t head_dim = n_embd / n_head;

        double bytes_per_elem_k = (double)ggml_type_size(gk) / (double)ggml_blck_size(gk);
        double bytes_per_elem_v = (double)ggml_type_size(gv) / (double)ggml_blck_size(gv);

        double kv_bytes_per_token_per_layer = n_head_kv * head_dim * (bytes_per_elem_k + bytes_per_elem_v);
        return static_cast<int64_t>(n_layer * kv_bytes_per_token_per_layer);
    });
}

int64_t argus_model_estimate_vram_bytes(const argus_model_t * model, int32_t context_length, int32_t type_k, int32_t type_v) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1LL, "argus_model_estimate_vram_bytes", [&]() -> int64_t {
        if (!model || !model->model || context_length < 0) {
            return -1;
        }

        int64_t kv_per_token = argus_model_kv_bytes_per_token(model, type_k, type_v);
        if (kv_per_token < 0) {
            return -1;
        }

        uint64_t model_bytes = argus_model_size(model);
        return static_cast<int64_t>(model_bytes) + (kv_per_token * static_cast<int64_t>(context_length));
    });
}

// =========================================================================
// Synchronized Context Operations
// =========================================================================

// Prunes invalidated KV cache tail if start_pos rolls back prior high-water mark
static void prune_kv_cache_if_rollback(struct llama_context * lctx, int32_t seq_id, int32_t start_pos) {
    if (!lctx) {
        return;
    }
    llama_memory_t mem = llama_get_memory(lctx);
    if (mem) {
        llama_pos cur_max = llama_memory_seq_pos_max(mem, seq_id);
        if (cur_max >= 0 && start_pos <= cur_max) {
            llama_memory_seq_rm(mem, seq_id, start_pos, -1);
        }
    }
}

// =========================================================================
// Cancellation Abort Flag Lifecycle & Management
// =========================================================================

argus_abort_flag_t * argus_abort_flag_create(void) {
    try {
        clear_last_error();
        auto * flag = new argus_abort_flag();
        flag->refs.store(1, std::memory_order_relaxed);
        flag->requested.store(false, std::memory_order_relaxed);
        return flag;
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
        return nullptr;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
        return nullptr;
    }
}

void argus_abort_flag_request(argus_abort_flag_t * flag) {
    if (flag) {
        flag->requested.store(true, std::memory_order_release);
    }
}

void argus_abort_flag_reset(argus_abort_flag_t * flag) {
    if (flag) {
        flag->requested.store(false, std::memory_order_release);
    }
}

bool argus_abort_flag_is_requested(const argus_abort_flag_t * flag) {
    if (!flag) {
        return false;
    }
    return flag->requested.load(std::memory_order_acquire);
}

bool argus_abort_flag_retain(argus_abort_flag_t * flag) {
    if (!flag) {
        return false;
    }
    uint32_t cur = flag->refs.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (flag->refs.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void argus_abort_flag_release(argus_abort_flag_t * flag) {
    if (!flag) {
        return;
    }
    if (flag->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete flag;
    }
}

namespace {
struct abort_flag_lease {
    argus_abort_flag_t * flag = nullptr;
    explicit abort_flag_lease(argus_abort_flag_t * f) {
        if (f && argus_abort_flag_retain(f)) {
            flag = f;
        }
    }
    ~abort_flag_lease() {
        if (flag) {
            argus_abort_flag_release(flag);
        }
    }
    argus_abort_flag_t * get() const { return flag; }
};
} // namespace

// Evaluates a sequence of tokens in chunks conforming to the model's native n_batch limit.
// Precondition: Caller must hold the context level synchronization mutex (ctx->mtx).
static int32_t decode_tokens_chunked(
    struct llama_context * lctx,
    const int32_t        * tokens,
    int32_t                n_tokens,
    int32_t                start_pos,
    int32_t                seq_id,
    bool                   request_logits,
    argus_abort_flag_t   * abort_flag,
    int32_t              * out_n_decoded = nullptr
) {
    if (out_n_decoded) {
        *out_n_decoded = 0;
    }

    if (!lctx || !tokens || n_tokens <= 0) {
        return 0;
    }

    int32_t n_batch = (int32_t)llama_n_batch(lctx);
    int32_t decoded = 0;

    int32_t max_chunk_size = std::min(n_batch, n_tokens);
    struct llama_batch batch = llama_batch_init(max_chunk_size, 0, 1);

    while (decoded < n_tokens) {
        int32_t chunk_size = std::min(n_batch, n_tokens - decoded);
        bool is_last_chunk = (decoded + chunk_size == n_tokens);
        // Only request logits on terminal token of final chunk if caller requested logits
        bool request_logits_chunk = request_logits && is_last_chunk;

        bool is_encoder_or_embeddings = llama_model_has_encoder(llama_get_model(lctx)) || (llama_pooling_type(lctx) != LLAMA_POOLING_TYPE_NONE);

        batch.n_tokens = chunk_size;

        for (int32_t i = 0; i < chunk_size; ++i) {
            batch.token[i]     = tokens[decoded + i];
            batch.pos[i]       = start_pos + decoded + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i]    = (is_encoder_or_embeddings || (request_logits_chunk && (i == chunk_size - 1))) ? 1 : 0;
        }

        if (argus_abort_flag_is_requested(abort_flag)) {
            llama_batch_free(batch);
            if (out_n_decoded) {
                *out_n_decoded = decoded;
            }
            return -2; // Aborted
        }

        int32_t result = 0;
        if (llama_model_has_encoder(llama_get_model(lctx))) {
            result = llama_encode(lctx, batch);
        } else {
            result = llama_decode(lctx, batch);
        }

        if (result != 0) {
            llama_batch_free(batch);
            if (out_n_decoded) {
                *out_n_decoded = decoded;
            }
            return result;
        }

        decoded += chunk_size;
        if (out_n_decoded) {
            *out_n_decoded = decoded;
        }
    }

    llama_batch_free(batch);
    return 0;
}

static void invalidate_seq_logits(argus_context_t * ctx, int32_t seq_id) {
    if (!ctx) {
        return;
    }
    if (seq_id < 0) {
        for (auto & slot : ctx->seq_samplers) {
            slot.has_logits = false;
            slot.last_logits_pos = -1;
        }
        ctx->last_decoded_seq_id = -1;
    } else if (seq_id < (int32_t)ctx->seq_samplers.size()) {
        auto & slot = ctx->seq_samplers[seq_id];
        slot.has_logits = false;
        slot.last_logits_pos = -1;
        if (ctx->last_decoded_seq_id == seq_id) {
            ctx->last_decoded_seq_id = -1;
        }
    }
}

static void rebuild_slot_chain_preserving_rng(argus_context_t * ctx, int32_t seq_id);

int32_t argus_decode_batch(argus_context_t * ctx, const argus_token_batch_t * batch_payload) {
    try {
        clear_last_error();
        if (!ctx || !batch_payload || !batch_payload->tokens || batch_payload->n_tokens <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid context or batch payload");
            return -1;
        }

        // Preflight cancellation check: if already aborted, abort immediately with zero mutations
        if (argus_abort_flag_is_requested(batch_payload->abort_flag)) {
            return -2;
        }

        abort_flag_lease lease(batch_payload->abort_flag);

        std::lock_guard<std::mutex> lock(ctx->mtx);

        int32_t seq_id = batch_payload->seq_id;
        if (seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "sequence ID out of range");
            return -1;
        }

        // Invalidate existing pending logits prior to state mutation
        invalidate_seq_logits(ctx, seq_id);

        // Query current max position in sequence to determine if a true KV rollback occurs
        llama_memory_t mem = llama_get_memory(ctx->ctx);
        llama_pos cur_max = mem ? llama_memory_seq_pos_max(mem, seq_id) : -1;
        bool is_kv_rollback = (cur_max >= 0 && batch_payload->start_pos <= cur_max);

        auto & slot = ctx->seq_samplers[seq_id];

        if (is_kv_rollback) {
            // Automagically prune invalidated KV cache tail across BOTH primary and speculative draft contexts
            prune_kv_cache_if_rollback(ctx->ctx, seq_id, batch_payload->start_pos);
            prune_kv_cache_if_rollback(ctx->draft_ctx, seq_id, batch_payload->start_pos);

            // Discard pending sample if targeted at or beyond rollback point
            bool pending_discarded = false;
            if (slot.pending.valid && slot.pending.kv_pos >= batch_payload->start_pos) {
                slot.pending = {};
                pending_discarded = true;
            }

            // Coordinate-decoupled rollback: prune only committed generated tokens matching kv_pos >= start_pos
            // Decoupled primed tokens (kv_pos == -1) and surviving prefix tokens (kv_pos < start_pos) are preserved.
            bool history_pruned = false;
            auto it = slot.history.begin();
            while (it != slot.history.end()) {
                if (it->kv_pos >= 0 && it->kv_pos >= batch_payload->start_pos) {
                    it = slot.history.erase(it);
                    history_pruned = true;
                } else {
                    ++it;
                }
            }

            if (history_pruned || pending_discarded) {
                rebuild_slot_chain_preserving_rng(ctx, seq_id);
            }

            int32_t n_decoded = 0;
            int32_t res = decode_tokens_chunked(
                ctx->ctx,
                batch_payload->tokens,
                batch_payload->n_tokens,
                batch_payload->start_pos,
                seq_id,
                batch_payload->request_logits,
                lease.get(),
                &n_decoded
            );

            // Closure of the state machine: commit successfully decoded rollback replacement tokens
            if (n_decoded >= 1) {
                for (int32_t i = 0; i < n_decoded; ++i) {
                    int32_t tok_pos = batch_payload->start_pos + i;
                    slot.history.push_back({batch_payload->tokens[i], tok_pos});
                    if (slot.chain) {
                        llama_sampler_accept(slot.chain, batch_payload->tokens[i]);
                    }
                }
            }

            if (res == 0) {
                ctx->last_decoded_seq_id = seq_id;
                slot.has_logits = batch_payload->request_logits;
                slot.last_logits_pos = batch_payload->request_logits ? (batch_payload->start_pos + batch_payload->n_tokens - 1) : -1;
            } else {
                invalidate_seq_logits(ctx, seq_id);
            }

            return res;
        }

        // Forward continuation / normal generation / prefill progression
        bool is_matching_continuation = slot.pending.valid &&
            (batch_payload->start_pos == slot.pending.kv_pos && batch_payload->tokens[0] == slot.pending.token);
        bool is_mismatched_override = slot.pending.valid && !is_matching_continuation;

        int32_t n_decoded = 0;
        int32_t res = decode_tokens_chunked(
            ctx->ctx,
            batch_payload->tokens,
            batch_payload->n_tokens,
            batch_payload->start_pos,
            seq_id,
            batch_payload->request_logits,
            lease.get(),
            &n_decoded
        );

        // Staged commit: only commit tokens that actually succeeded in native decode
        if (n_decoded >= 1) {
            if (is_matching_continuation) {
                // Normal continuation commit: token 0 was already accepted into chain during sampling.
                // Move from pending to committed history without resetting chain or RNG.
                slot.history.push_back({slot.pending.token, slot.pending.kv_pos});
                slot.pending = {};

                // If caller passed additional tokens in continuation batch, accept and commit them
                for (int32_t i = 1; i < n_decoded; ++i) {
                    int32_t tok_pos = batch_payload->start_pos + i;
                    slot.history.push_back({batch_payload->tokens[i], tok_pos});
                    if (slot.chain) {
                        llama_sampler_accept(slot.chain, batch_payload->tokens[i]);
                    }
                }
            } else if (is_mismatched_override) {
                // Mismatched token decoded instead of sampled pending token.
                // Discard pending, rebuild chain to evict speculative sample, and accept decoded batch prefix.
                slot.pending = {};
                rebuild_slot_chain_preserving_rng(ctx, seq_id);

                for (int32_t i = 0; i < n_decoded; ++i) {
                    int32_t tok_pos = batch_payload->start_pos + i;
                    slot.history.push_back({batch_payload->tokens[i], tok_pos});
                    if (slot.chain) {
                        llama_sampler_accept(slot.chain, batch_payload->tokens[i]);
                    }
                }
            }
        }

        if (res == 0) {
            ctx->last_decoded_seq_id = seq_id;
            slot.has_logits = batch_payload->request_logits;
            slot.last_logits_pos = batch_payload->request_logits ? (batch_payload->start_pos + batch_payload->n_tokens - 1) : -1;
        } else {
            invalidate_seq_logits(ctx, seq_id);
        }

        return res;
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
        return -1;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
        return -1;
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during batch decode");
        return -1;
    }
}

int32_t argus_get_embeddings(argus_context_t * ctx, int32_t seq_id, float * out_embeddings, int32_t max_floats) {
    if (!ctx || !out_embeddings || max_floats <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    float * embd = llama_get_embeddings_seq(ctx->ctx, seq_id);
    if (!embd) {
        embd = llama_get_embeddings_ith(ctx->ctx, -1);
    }
    if (!embd) {
        embd = llama_get_embeddings(ctx->ctx);
    }
    if (!embd) {
        return -1;
    }

    int32_t n_embd = llama_model_n_embd(ctx->model_ref->model);
    if (n_embd > max_floats) {
        return -2;
    }

    std::memcpy(out_embeddings, embd, n_embd * sizeof(float));
    return n_embd;
}

// =========================================================================
// Zero-Allocation Persistent Sampler Pipeline
// =========================================================================

static bool sampler_params_equal(const argus_sampler_params_t & a, const argus_sampler_params_t & b) {
    return a.temperature == b.temperature &&
           a.repeat_penalty == b.repeat_penalty &&
           a.repeat_last_n == b.repeat_last_n &&
           a.frequency_penalty == b.frequency_penalty &&
           a.presence_penalty == b.presence_penalty &&
           a.top_p == b.top_p &&
           a.min_p == b.min_p &&
           a.top_k == b.top_k &&
           a.dry_multiplier == b.dry_multiplier &&
           a.dry_base == b.dry_base &&
           a.dry_allowed_length == b.dry_allowed_length &&
           a.dry_penalty_last_n == b.dry_penalty_last_n &&
           a.seed == b.seed;
}

static bool logit_biases_equal(const std::vector<argus_logit_bias_t> & cached, const argus_logit_bias_t * biases, int32_t bias_count) {
    if ((int32_t)cached.size() != bias_count) {
        return false;
    }
    if (bias_count == 0) {
        return true;
    }
    if (!biases) {
        return false;
    }
    for (int32_t i = 0; i < bias_count; ++i) {
        if (cached[i].token != biases[i].token || cached[i].bias != biases[i].bias) {
            return false;
        }
    }
    return true;
}

static struct llama_sampler * build_sampler_chain(
    const argus_context_t        * ctx,
    const argus_sampler_params_t * sparams,
    const argus_logit_bias_t     * biases,
    int32_t                        bias_count,
    struct llama_sampler         * existing_dist = nullptr
) {
    const struct llama_vocab * vocab = (ctx->model_ref && ctx->model_ref->vocab) 
        ? ctx->model_ref->vocab 
        : llama_model_get_vocab(llama_get_model(ctx->ctx));
    int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 32000;

    struct llama_sampler_chain_params scparams = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(scparams);

    // 1. Logit biases (applied first to shift token logits)
    if (biases && bias_count > 0 && vocab) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, bias_count, (const llama_logit_bias *)biases));
    }

    // 2. Penalties (repetition, frequency, presence)
    if (sparams->repeat_penalty > 1.0f || sparams->frequency_penalty != 0.0f || sparams->presence_penalty != 0.0f) {
        int32_t last_n = (sparams->repeat_last_n > 0) ? sparams->repeat_last_n : 64;
        float repeat_pen = (sparams->repeat_penalty > 0.0f) ? sparams->repeat_penalty : 1.0f;
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, last_n, repeat_pen, sparams->frequency_penalty, sparams->presence_penalty));
    }

    // 3. DRY Sampler
    if (sparams->dry_multiplier > 0.0f && vocab) {
        float dry_base = (sparams->dry_base > 0.0f) ? sparams->dry_base : 1.75f;
        int32_t dry_allowed_length = (sparams->dry_allowed_length > 0) ? sparams->dry_allowed_length : 2;
        int32_t dry_penalty_last_n = (sparams->dry_penalty_last_n != 0) ? sparams->dry_penalty_last_n : -1;
        llama_sampler_chain_add(chain, llama_sampler_init_dry(vocab, sparams->dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, nullptr, 0));
    }

    // 4. Top-K
    if (sparams->top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sparams->top_k));
    }

    // 5. Top-P
    if (sparams->top_p > 0.0f && sparams->top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sparams->top_p, 1));
    }

    // 6. Min-P
    if (sparams->min_p > 0.0f && sparams->min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(sparams->min_p, 1));
    }

    // 7. Temperature & Stochastic Distribution / Greedy Selection
    if (sparams->temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(sparams->temperature));
        if (existing_dist) {
            llama_sampler_chain_add(chain, existing_dist);
        } else {
            uint32_t dist_seed = (sparams->seed == 0xFFFFFFFF)
                ? LLAMA_DEFAULT_SEED
                : sparams->seed;
            llama_sampler_chain_add(chain, llama_sampler_init_dist(dist_seed));
        }
    }

    return chain;
}

static struct llama_sampler * clone_active_dist_sampler(struct llama_sampler * chain) {
    if (!chain) {
        return nullptr;
    }
    int chain_len = llama_sampler_chain_n(chain);
    if (chain_len > 0) {
        struct llama_sampler * last_sampler = llama_sampler_chain_get(chain, chain_len - 1);
        if (last_sampler) {
            return llama_sampler_clone(last_sampler);
        }
    }
    return nullptr;
}

static void rebuild_slot_chain_preserving_rng(argus_context_t * ctx, int32_t seq_id) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return;
    }
    auto & slot = ctx->seq_samplers[seq_id];
    if (!slot.has_cached_chain) {
        return;
    }

    struct llama_sampler * preserved_dist = nullptr;
    if (slot.chain && slot.cached_sparams.temperature > 0.0f) {
        preserved_dist = clone_active_dist_sampler(slot.chain);
    }

    if (slot.chain) {
        llama_sampler_free(slot.chain);
        slot.chain = nullptr;
    }

    slot.chain = build_sampler_chain(
        ctx,
        &slot.cached_sparams,
        slot.cached_biases.empty() ? nullptr : slot.cached_biases.data(),
        (int32_t)slot.cached_biases.size(),
        preserved_dist
    );

    if (slot.chain && !slot.history.empty()) {
        for (const auto & entry : slot.history) {
            llama_sampler_accept(slot.chain, entry.token);
        }
    }
    if (slot.chain && slot.pending.valid) {
        llama_sampler_accept(slot.chain, slot.pending.token);
    }
}

bool discard_slot_pending_preserving_rng(argus_context_t * ctx, int32_t seq_id) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return false;
    }
    auto & slot = ctx->seq_samplers[seq_id];
    if (!slot.pending.valid) {
        return false;
    }
    slot.pending = {};
    rebuild_slot_chain_preserving_rng(ctx, seq_id);
    return true;
}

int32_t argus_sampler_discard_pending(argus_context_t * ctx, int32_t seq_id) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx->mtx);
    return discard_slot_pending_preserving_rng(ctx, seq_id) ? 1 : 0;
}

static struct llama_sampler * get_or_update_slot_sampler(
    argus_context_t              * ctx,
    int32_t                        seq_id,
    const argus_sampler_params_t * sparams,
    const argus_logit_bias_t     * biases,
    int32_t                        bias_count
) {
    if (seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return nullptr;
    }

    auto & slot = ctx->seq_samplers[seq_id];

    if (!slot.has_cached_chain ||
        !sampler_params_equal(slot.cached_sparams, *sparams) ||
        !logit_biases_equal(slot.cached_biases, biases, bias_count)) {

        struct llama_sampler * preserved_dist = nullptr;

        // If seed is unchanged and temperature > 0, preserve active distribution sampler RNG state
        if (slot.chain && slot.has_cached_chain &&
            slot.cached_sparams.temperature > 0.0f && sparams->temperature > 0.0f &&
            slot.cached_sparams.seed == sparams->seed) {
            preserved_dist = clone_active_dist_sampler(slot.chain);
        }

        if (slot.chain) {
            llama_sampler_free(slot.chain);
            slot.chain = nullptr;
        }

        slot.chain = build_sampler_chain(ctx, sparams, biases, bias_count, preserved_dist);
        slot.cached_sparams = *sparams;
        if (biases && bias_count > 0) {
            slot.cached_biases.assign(biases, biases + bias_count);
        } else {
            slot.cached_biases.clear();
        }
        slot.has_cached_chain = true;

        // Replay retained sequence history into newly constructed chain
        if (slot.chain && !slot.history.empty()) {
            for (const auto & entry : slot.history) {
                llama_sampler_accept(slot.chain, entry.token);
            }
        }
        if (slot.chain && slot.pending.valid) {
            llama_sampler_accept(slot.chain, slot.pending.token);
        }
    }

    return slot.chain;
}

int32_t argus_sample_token_ext(
    argus_context_t              * ctx,
    int32_t                        seq_id,
    const argus_sampler_params_t * sparams,
    const argus_logit_bias_t     * biases,
    int32_t                        bias_count
) {
    if (!ctx || !ctx->ctx || !sparams) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    if (seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }

    auto & slot = ctx->seq_samplers[seq_id];

    // Logits validation: Must be the sequence evaluated last, with logits requested and available
    if (ctx->last_decoded_seq_id != seq_id || !slot.has_logits) {
        return -2; // Logits unavailable for target sequence
    }

    struct llama_sampler * sampler = get_or_update_slot_sampler(ctx, seq_id, sparams, biases, bias_count);
    if (!sampler) {
        return -1;
    }

    // Sample from the last logits row (-1). llama_sampler_sample accepts the token internally.
    int32_t token = llama_sampler_sample(sampler, ctx->ctx, -1);

    // Record token into sequence pending sample tagged with its expected KV cache position
    int32_t assigned_kv_pos = (slot.last_logits_pos >= 0) ? (slot.last_logits_pos + 1) : -1;
    slot.pending.token = token;
    slot.pending.kv_pos = assigned_kv_pos;
    slot.pending.valid = true;

    // Consume logits
    slot.has_logits = false;
    slot.last_logits_pos = -1;

    return token;
}

int32_t argus_sample_token(argus_context_t * ctx, int32_t seq_id, float temperature, float repeat_penalty) {
    argus_sampler_params_t sparams = {};
    sparams.temperature    = temperature;
    sparams.repeat_penalty = repeat_penalty;
    sparams.repeat_last_n  = 64;
    sparams.top_p          = 1.0f;
    sparams.min_p          = 0.0f;
    sparams.top_k          = 0;
    sparams.seed           = 0xFFFFFFFF;

    return argus_sample_token_ext(ctx, seq_id, &sparams, nullptr, 0);
}

int32_t argus_sample_token_with_bias(
    argus_context_t          * ctx, 
    int32_t                    seq_id, 
    float                      temperature, 
    float                      repeat_penalty, 
    const argus_logit_bias_t * biases, 
    int32_t                    bias_count
) {
    argus_sampler_params_t sparams = {};
    sparams.temperature    = temperature;
    sparams.repeat_penalty = repeat_penalty;
    sparams.repeat_last_n  = 64;
    sparams.top_p          = 1.0f;
    sparams.min_p          = 0.0f;
    sparams.top_k          = 0;
    sparams.seed           = 0xFFFFFFFF;

    return argus_sample_token_ext(ctx, seq_id, &sparams, biases, bias_count);
}

int32_t argus_sampler_reset(argus_context_t * ctx, int32_t seq_id) {
    if (!ctx) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx->mtx);

    invalidate_seq_logits(ctx, seq_id);

    if (seq_id < 0) {
        // Reset all sequence slots
        for (auto & slot : ctx->seq_samplers) {
            slot.history.clear();
            slot.pending = {};
            if (slot.chain) {
                llama_sampler_reset(slot.chain);
            }
        }
        return 0;
    }

    if (seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }

    auto & slot = ctx->seq_samplers[seq_id];
    slot.history.clear();
    slot.pending = {};
    if (slot.chain) {
        llama_sampler_reset(slot.chain);
    }
    return 0;
}

int32_t argus_sampler_prime(argus_context_t * ctx, int32_t seq_id, const int32_t * tokens, int32_t n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0 || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx->mtx);

    // If a sample is pending, discard it and rebuild before priming external tokens
    discard_slot_pending_preserving_rng(ctx, seq_id);

    auto & slot = ctx->seq_samplers[seq_id];
    for (int32_t i = 0; i < n_tokens; ++i) {
        slot.history.push_back({tokens[i], -1}); // -1: Decoupled primed token
        if (slot.chain) {
            llama_sampler_accept(slot.chain, tokens[i]);
        }
    }
    return 0;
}

int32_t argus_sampler_truncate(argus_context_t * ctx, int32_t seq_id, int32_t new_length) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size() || new_length < 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx->mtx);

    auto & slot = ctx->seq_samplers[seq_id];
    bool pending_was_valid = slot.pending.valid;
    slot.pending = {};
    if (new_length < (int32_t)slot.history.size() || pending_was_valid) {
        if (new_length < (int32_t)slot.history.size()) {
            slot.history.resize((size_t)new_length);
        }
        rebuild_slot_chain_preserving_rng(ctx, seq_id);
    }
    return 0;
}

int32_t argus_sampler_get_history_count(const argus_context_t * ctx, int32_t seq_id) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(const_cast<argus_context_t *>(ctx)->mtx);
    return (int32_t)ctx->seq_samplers[seq_id].history.size();
}

int32_t argus_sampler_has_pending(const argus_context_t * ctx, int32_t seq_id) {
    if (!ctx || seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(const_cast<argus_context_t *>(ctx)->mtx);
    return ctx->seq_samplers[seq_id].pending.valid ? 1 : 0;
}

void argus_kv_cache_clear_slot(argus_context_t * ctx, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    // Direct binding route targeting unmanaged sequence tracking cells
    if (ctx->ctx) {
        llama_memory_seq_rm(llama_get_memory(ctx->ctx), seq_id, p0, p1);
    }
    if (ctx->draft_ctx) {
        llama_memory_seq_rm(llama_get_memory(ctx->draft_ctx), seq_id, p0, p1);
    }

    // Invalidate logits across sequence slots
    invalidate_seq_logits(ctx, seq_id);

    // Sequence-specific sampler reset / coordinate-linked pruning
    if (seq_id < 0) {
        for (auto & slot : ctx->seq_samplers) {
            slot.history.clear();
            slot.pending = {};
            if (slot.chain) {
                llama_sampler_reset(slot.chain);
            }
        }
    } else if (seq_id < (int32_t)ctx->seq_samplers.size()) {
        auto & slot = ctx->seq_samplers[seq_id];
        if (p0 <= 0 && p1 < 0) {
            // Full clear
            slot.history.clear();
            slot.pending = {};
            if (slot.chain) {
                llama_sampler_reset(slot.chain);
            }
        } else if (p0 >= 0) {
            // Partial clear: prune entries where kv_pos >= p0 && (p1 < 0 || kv_pos < p1) && kv_pos >= 0
            bool pending_discarded = false;
            if (slot.pending.valid && slot.pending.kv_pos >= p0 && (p1 < 0 || slot.pending.kv_pos < p1)) {
                slot.pending = {};
                pending_discarded = true;
            }
            bool history_pruned = false;
            auto it = slot.history.begin();
            while (it != slot.history.end()) {
                if (it->kv_pos >= 0 && it->kv_pos >= p0 && (p1 < 0 || it->kv_pos < p1)) {
                    it = slot.history.erase(it);
                    history_pruned = true;
                } else {
                    ++it;
                }
            }
            if (history_pruned || pending_discarded) {
                rebuild_slot_chain_preserving_rng(ctx, seq_id);
            }
        }
    }
}

int32_t argus_kv_cache_seq_pos_max(const argus_context_t * ctx, int32_t seq_id) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_kv_cache_seq_pos_max", [&]() -> int32_t {
        if (!ctx || !ctx->ctx) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(const_cast<argus_context_t *>(ctx)->mtx);
        llama_memory_t mem = llama_get_memory(ctx->ctx);
        return mem ? (int32_t)llama_memory_seq_pos_max(mem, seq_id) : -1;
    });
}

int32_t argus_kv_cache_seq_pos_min(const argus_context_t * ctx, int32_t seq_id) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_kv_cache_seq_pos_min", [&]() -> int32_t {
        if (!ctx || !ctx->ctx) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(const_cast<argus_context_t *>(ctx)->mtx);
        llama_memory_t mem = llama_get_memory(ctx->ctx);
        return mem ? (int32_t)llama_memory_seq_pos_min(mem, seq_id) : -1;
    });
}

// =========================================================================
// TTS: Speech Synthesis
// =========================================================================

int32_t argus_synthesize_speech_n(
    argus_context_t * ctx,
    const argus_model_t * wavtokenizer_model,
    const char * text,
    size_t text_len,
    int32_t voice_seed,
    float * out_pcm,
    int32_t max_samples,
    float * workspace,
    int64_t workspace_size_floats) {

    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_synthesize_speech_n", [&]() -> int32_t {
        if (!ctx || !wavtokenizer_model || !text || text_len == 0 || text_len > (size_t)INT32_MAX || !out_pcm || max_samples <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid arguments to argus_synthesize_speech_n");
            return -1;
        }

    std::lock_guard<std::mutex> lock(ctx->mtx);
    (void)voice_seed;

    // Clear KV cache for sequence 0 to prevent position conflicts on retries
    llama_memory_seq_rm(llama_get_memory(ctx->ctx), 0, -1, -1);

    const struct llama_vocab * vocab = ctx->model_ref->vocab;

    // Default speaker setup matching en_male_1
    std::string audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";
    std::string audio_data = "<|audio_start|>\n"
"the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>\n"
"overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>\n"
"package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>\n"
"from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>\n"
"just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>\n"
"two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>\n"
"people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>\n"
"is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>\n"
"pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>\n"
"remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>\n"
"sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>\n"
"i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>\n"
"have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>\n"
"some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>\n"
"critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>\n"
"about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>\n"
"some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>\n"
"of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>\n"
"the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>\n"
"gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>\n"
"aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>\n"
"but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>\n"
"its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>\n"
"still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>\n"
"really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>\n"
"enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>\n"
"and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>\n"
"it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>\n"
"looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>\n"
"lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>\n";

    std::vector<llama_token> prompt_inp;

    // init prompt with BOS
    tokenize_to_vector(vocab, "<|im_start|>\n", prompt_inp, true);

    // add speaker prompt prefix
    tokenize_to_vector(vocab, audio_text, prompt_inp, false);

    // tokenize input text
    std::string text_str(text, text_len);
    std::string prompt_clean = process_text_for_outetts(text_str);
    tokenize_to_vector(vocab, prompt_clean, prompt_inp, false);

    tokenize_to_vector(vocab, "<|text_end|>\n", prompt_inp, false);

    // add default speaker codes data
    tokenize_to_vector(vocab, audio_data, prompt_inp, false);

    // run decode batch for initial prompt evaluation using DRY chunked helper
    if (decode_tokens_chunked(ctx->ctx, prompt_inp.data(), (int32_t)prompt_inp.size(), 0, 0, true, nullptr) != 0) {
        return -1;
    }

    std::vector<llama_token> generated_codes;
    int n_past = prompt_inp.size();
    int n_decode = 0;
    const int n_predict = 4096;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sparams);
    // Apply repeating penalty and temperature
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(voice_seed > 0 ? (uint32_t)voice_seed : LLAMA_DEFAULT_SEED));

    struct llama_batch batch = llama_batch_init(1, 0, 1);

    while (n_decode <= n_predict) {
        llama_token new_token_id = llama_sampler_sample(sampler, ctx->ctx, -1);
        // Note: llama_sampler_sample already accepts token internally

        generated_codes.push_back(new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        n_decode++;
        n_past++;

        if (llama_decode(ctx->ctx, batch) != 0) {
            break;
        }
    }
    llama_sampler_free(sampler);
    llama_batch_free(batch);

    // Extract audio codes (filter range [151672, 155772])
    std::vector<llama_token> vocoder_codes;
    for (auto t : generated_codes) {
        if (t >= 151672 && t <= 155772) {
            vocoder_codes.push_back(t - 151672);
        }
    }

    const int n_codes = vocoder_codes.size();
    if (n_codes == 0) {
        return 0;
    }

    // Check if we need to initialize or update the persistent vocoder context
    if (ctx->vocoder_ctx && ctx->vocoder_model_ref != wavtokenizer_model) {
        llama_free(ctx->vocoder_ctx);
        ctx->vocoder_ctx = nullptr;
        if (ctx->vocoder_model_ref) {
            argus_model_release(const_cast<argus_model_t *>(ctx->vocoder_model_ref));
            ctx->vocoder_model_ref = nullptr;
        }
    }

    if (!ctx->vocoder_ctx) {
        struct llama_context_params vocoder_cparams = llama_context_default_params();
        vocoder_cparams.n_ctx    = 4096;
        vocoder_cparams.n_batch  = 4096;
        vocoder_cparams.n_ubatch = 512;
        vocoder_cparams.n_threads = 4;
        vocoder_cparams.embeddings = true;

        ctx->vocoder_ctx = llama_init_from_model(wavtokenizer_model->model, vocoder_cparams);
        if (!ctx->vocoder_ctx) {
            return -1;
        }
        argus_model_retain(const_cast<argus_model_t *>(wavtokenizer_model));
        ctx->vocoder_model_ref = wavtokenizer_model;
    }

    struct llama_context * active_vocoder_ctx = ctx->vocoder_ctx;
    bool is_temporary_vocoder_ctx = false;

    if (n_codes + 16 > 4096) {
        struct llama_context_params vocoder_cparams = llama_context_default_params();
        vocoder_cparams.n_ctx    = n_codes + 16;
        vocoder_cparams.n_batch  = n_codes + 16;
        vocoder_cparams.n_ubatch = 512;
        vocoder_cparams.n_threads = 4;
        vocoder_cparams.embeddings = true;

        active_vocoder_ctx = llama_init_from_model(wavtokenizer_model->model, vocoder_cparams);
        if (!active_vocoder_ctx) {
            return -1;
        }
        is_temporary_vocoder_ctx = true;
    } else {
        llama_memory_seq_rm(llama_get_memory(active_vocoder_ctx), 0, -1, -1);
    }

    struct llama_batch vocoder_batch = llama_batch_init(n_codes, 0, 1);
    vocoder_batch.n_tokens = n_codes;
    for (int i = 0; i < n_codes; ++i) {
        vocoder_batch.token[i]     = vocoder_codes[i];
        vocoder_batch.pos[i]       = i;
        vocoder_batch.n_seq_id[i]  = 1;
        vocoder_batch.seq_id[i][0] = 0;
        vocoder_batch.logits[i]    = false;
    }

    if (llama_decode(active_vocoder_ctx, vocoder_batch) != 0) {
        llama_batch_free(vocoder_batch);
        if (is_temporary_vocoder_ctx) {
            llama_free(active_vocoder_ctx);
        }
        return -1;
    }
    llama_synchronize(active_vocoder_ctx);

    const int n_embd = llama_model_n_embd_out(wavtokenizer_model->model);
    const float * embd = llama_get_embeddings(active_vocoder_ctx);
    if (!embd) {
        llama_batch_free(vocoder_batch);
        if (is_temporary_vocoder_ctx) {
            llama_free(active_vocoder_ctx);
        }
        return -1;
    }

    // Reconstruct spectral audio floats using the zero-allocation workspace allocator
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;

    int n_spec = n_embd * n_codes;

    // Single flat memory buffer allocation (zero allocation churn)
    size_t total_floats = n_spec * 3 + (n_codes * n_fft) * 2 + n_out * 2;
    // Add spaces for irfft scratchpad arrays:
    // scratch_real_inp: 641, scratch_imag_inp: 641, scratch_real_out: 1280, scratch_imag_out: 1280
    size_t scratch_floats = 641 * 2 + 1280 * 2;
    size_t required_floats = total_floats + scratch_floats;

    if (!workspace || workspace_size_floats < (int64_t)required_floats) {
        llama_batch_free(vocoder_batch);
        if (is_temporary_vocoder_ctx) {
            llama_free(active_vocoder_ctx);
        }
        return -(int32_t)required_floats; // Signal workspace resize required
    }

    float * E = workspace;
    float * S = E + n_spec;
    float * ST = S + n_spec;
    float * res = ST + n_spec;
    float * hann2 = res + (n_codes * n_fft);
    float * audio = hann2 + (n_codes * n_fft);
    float * env = audio + n_out;

    float * scr_real_inp = env + n_out;
    float * scr_imag_inp = scr_real_inp + 641;
    float * scr_real_out = scr_imag_inp + 641;
    float * scr_imag_out = scr_real_out + 1280;

    alignas(64) float hann[1280];
    fill_hann_window(n_fft, true, hann);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k * n_codes + l] = embd[l * n_embd + k];
        }
    }

    for (int k = 0; k < n_embd / 2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k)*n_codes + l];
            float phi = E[(k + n_embd / 2)*n_codes + l];

            mag = expf(mag);
            if (mag > 1e2f) {
                mag = 1e2f;
            }
            S[2 * (k * n_codes + l) + 0] = mag * cosf(phi);
            S[2 * (k * n_codes + l) + 1] = mag * sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd / 2; ++k) {
            ST[l * n_embd + 2 * k + 0] = S[2 * (k * n_codes + l) + 0];
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1];
        }
    }

    // Run IRFFT on each segment using the pre-allocated scratch workspace
    for (int l = 0; l < n_codes; ++l) {
        irfft(n_fft, ST + l * n_embd, res + l * n_fft, scr_real_inp, scr_imag_inp, scr_real_out, scr_imag_out);
        for (int j = 0; j < n_fft; ++j) {
            res[l * n_fft + j] *= hann[j];
            hann2[l * n_fft + j] = hann[j] * hann[j];
        }
    }

    // Fold overlap-add matrices
    fold(res, n_codes * n_fft, n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_codes * n_fft, n_out, n_win, n_hop, n_pad, env);

    int64_t final_size = n_out - 2 * n_pad;
    int32_t count_to_write = (max_samples < final_size) ? max_samples : (int32_t)final_size;

    for (int32_t i = 0; i < count_to_write; ++i) {
        out_pcm[i] = audio[i] / env[i];
    }

    // Zero out first 0.25 seconds (same as in llama.cpp tools/tts/tts.cpp line 1078)
    int32_t silence_samples = 24000 / 4;
    for (int32_t i = 0; i < silence_samples && i < count_to_write; ++i) {
        out_pcm[i] = 0.0f;
    }

    llama_batch_free(vocoder_batch);
    if (is_temporary_vocoder_ctx) {
        llama_free(active_vocoder_ctx);
    }

    return count_to_write;
    });
}

int32_t argus_synthesize_speech(
    argus_context_t * ctx,
    const argus_model_t * wavtokenizer_model,
    const char * text,
    int32_t voice_seed,
    float * out_pcm,
    int32_t max_samples,
    float * workspace,
    int64_t workspace_size_floats) {
    if (!text) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "text pointer is NULL");
        return -1;
    }
    return argus_synthesize_speech_n(
        ctx, wavtokenizer_model, text, std::strlen(text), voice_seed, out_pcm, max_samples, workspace, workspace_size_floats);
}

} // extern "C"