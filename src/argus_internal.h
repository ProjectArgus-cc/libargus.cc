#ifndef ARGUS_INTERNAL_H
#define ARGUS_INTERNAL_H

#include "libargus.h"
#include "llama.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>
#include <limits>
#include <exception>
#include <new>

#ifdef ARGUS_TESTING
extern "C" bool argus_test_should_fail_batch_alloc(int32_t step) noexcept;
#endif

struct argus_owned_batch {
    struct llama_batch                  raw = {};
    std::unique_ptr<llama_token[]>      token_buf;
    std::unique_ptr<float[]>            embd_buf;
    std::unique_ptr<llama_pos[]>        pos_buf;
    std::unique_ptr<int32_t[]>          n_seq_id_buf;
    std::unique_ptr<llama_seq_id *[]>   seq_id_ptrs;
    std::unique_ptr<llama_seq_id[]>     seq_id_pool;
    std::unique_ptr<int8_t[]>           logits_buf;

    argus_owned_batch() noexcept = default;
    ~argus_owned_batch() noexcept { reset(); }

    argus_owned_batch(const argus_owned_batch &) = delete;
    argus_owned_batch & operator=(const argus_owned_batch &) = delete;

    argus_owned_batch(argus_owned_batch &&) noexcept = default;
    argus_owned_batch & operator=(argus_owned_batch &&) noexcept = default;

    void reset() noexcept {
        raw = {};
        token_buf.reset();
        embd_buf.reset();
        pos_buf.reset();
        n_seq_id_buf.reset();
        seq_id_ptrs.reset();
        seq_id_pool.reset();
        logits_buf.reset();
    }

    bool init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) noexcept {
        reset();
        if (n_tokens_alloc <= 0 || n_seq_max <= 0 || embd < 0) {
            return false;
        }
        try {
            const size_t n_tok = static_cast<size_t>(n_tokens_alloc);
            const size_t n_seq = static_cast<size_t>(n_seq_max);
            if (n_tok > std::numeric_limits<size_t>::max() / n_seq) {
                return false;
            }
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(1)) throw std::bad_alloc();
#endif
            if (embd > 0) {
                const size_t n_embd = static_cast<size_t>(embd);
                if (n_tok > std::numeric_limits<size_t>::max() / n_embd) {
                    return false;
                }
                embd_buf.reset(new float[n_tok * n_embd]());
            } else {
                token_buf.reset(new llama_token[n_tok]());
            }
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(2)) throw std::bad_alloc();
#endif
            pos_buf.reset(new llama_pos[n_tok]());
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(3)) throw std::bad_alloc();
#endif
            n_seq_id_buf.reset(new int32_t[n_tok]());
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(4)) throw std::bad_alloc();
#endif
            seq_id_ptrs.reset(new llama_seq_id *[n_tok + 1]());
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(5)) throw std::bad_alloc();
#endif
            seq_id_pool.reset(new llama_seq_id[n_tok * n_seq]());
            for (size_t i = 0; i < n_tok; ++i) {
                seq_id_ptrs[i] = &seq_id_pool[i * n_seq];
            }
            seq_id_ptrs[n_tok] = nullptr;
#ifdef ARGUS_TESTING
            if (argus_test_should_fail_batch_alloc(6)) throw std::bad_alloc();
#endif
            logits_buf.reset(new int8_t[n_tok]());

            raw.n_tokens = 0;
            raw.token    = token_buf.get();
            raw.embd     = embd_buf.get();
            raw.pos      = pos_buf.get();
            raw.n_seq_id = n_seq_id_buf.get();
            raw.seq_id   = seq_id_ptrs.get();
            raw.logits   = logits_buf.get();
            return true;
        } catch (...) {
            reset();
            return false;
        }
    }
};

struct argus_model {
    std::atomic<uint32_t>          refs{1};
    struct llama_model           * model;
    const  struct llama_vocab    * vocab;
};

struct argus_abort_flag {
    std::atomic<uint32_t>          refs{1};
    std::atomic<bool>              requested{false};
};

// Thread-local diagnostic state helpers
void set_last_error(argus_error_code_t code, const char * msg);
void clear_last_error();

// C++17 Zero-overhead noexcept exception barrier templates
template <typename F, typename R>
inline R argus_guard(argus_error_code_t category, R failure_val, const char * op, F && fn) noexcept {
    try {
        clear_last_error();
        return fn();
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(category, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, op ? op : "unknown exception");
    }
    return failure_val;
}

template <typename F>
inline void argus_guard_void(const char * op, F && fn) noexcept {
    try {
        clear_last_error();
        fn();
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, op ? op : "unknown exception");
    }
}

// Backend active resource tracking
void argus_backend_resource_inc();
void argus_backend_resource_dec();

struct argus_history_entry {
    llama_token token;   // 4 bytes: vocabulary token ID
    int32_t     kv_pos;  // 4 bytes: evaluated KV cache position (-1 for decoupled/primed tokens)
};

struct argus_pending_sample {
    llama_token token   = -1;
    int32_t     kv_pos  = -1;
    bool        valid   = false;
};

struct argus_seq_sampler {
    struct llama_sampler             * chain              = nullptr;
    argus_sampler_params_t             cached_sparams     = {};
    std::vector<argus_logit_bias_t>    cached_biases;
    std::vector<argus_history_entry>   history;           // Pre-reserved to context_length (primed + committed)
    argus_pending_sample               pending;           // Active sample accepted into chain, awaiting decode
    int32_t                            last_logits_pos    = -1;
    bool                               has_logits         = false;
    bool                               has_embeddings     = false;
    uint64_t                           embedding_generation = 0;
    bool                               has_cached_chain   = false;
};

struct argus_context {
    struct llama_context           * ctx;
    struct llama_context           * draft_ctx;
    argus_model                    * model_ref;
    argus_model                    * draft_model_ref;
    int32_t                          context_length = 0;
    int32_t                          spec_draft_n_max;
    bool                             enable_draft_mtp;
    struct llama_context           * vocoder_ctx;
    const argus_model              * vocoder_model_ref;
    std::mutex                       mtx;                 // per-context concurrency guard

    // Context-level scratch batch for zero-allocation steady-state decode
    argus_owned_batch                batch_scratch_owner;
    struct llama_batch               batch_scratch      = {};
    bool                             has_batch_scratch  = false;

    // Sequence-isolated persistent sampler state slots
    std::vector<argus_seq_sampler>   seq_samplers;
    int32_t                          last_decoded_seq_id = -1;
    uint64_t                         embedding_generation = 0;
};

// Internal helper to discard a sequence slot's pending sample and rebuild its chain
extern "C" bool discard_slot_pending_preserving_rng(argus_context_t * ctx, int32_t seq_id);

// Internal helper to discard pending sample and prune committed history entries at kv_pos >= start_pos,
// rebuilding the slot sampler chain at most once while preserving RNG state.
extern "C" bool argus_prune_slot_from_pos_preserving_rng(argus_context_t * ctx, int32_t seq_id, int32_t start_pos);

// Internal helper to prune KV cache tail [start_pos, -1) across draft and primary contexts on rollback.
extern "C" bool argus_prune_context_kv_if_rollback(argus_context_t * ctx, int32_t seq_id, int32_t start_pos);

// Pure UTF-8 boundary truncation helper: backtracks only if a trailing multi-byte sequence is incomplete.
inline int32_t argus_utf8_truncate_incomplete_tail(char * out_text, int32_t total_written_chars) noexcept {
    if (!out_text || total_written_chars <= 0) {
        return 0;
    }
    int32_t idx = total_written_chars - 1;
    while (idx >= 0 && (static_cast<uint8_t>(out_text[idx]) & 0xC0) == 0x80) {
        idx--;
    }
    if (idx < 0) {
        out_text[0] = '\0';
        return 0;
    }
    uint8_t lead = static_cast<uint8_t>(out_text[idx]);
    if ((lead & 0x80) != 0) {
        int32_t expected_len = 0;
        if ((lead & 0xE0) == 0xC0) expected_len = 2;
        else if ((lead & 0xF0) == 0xE0) expected_len = 3;
        else if ((lead & 0xF8) == 0xF0) expected_len = 4;
        else {
            out_text[idx] = '\0';
            return idx;
        }
        if (total_written_chars - idx < expected_len) {
            out_text[idx] = '\0';
            return idx;
        }
    }
    return total_written_chars;
}

#endif // ARGUS_INTERNAL_H

