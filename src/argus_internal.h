#ifndef ARGUS_INTERNAL_H
#define ARGUS_INTERNAL_H

#include "libargus.h"
#include "llama.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <exception>
#include <new>

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
    bool                               has_cached_chain   = false;
};

struct argus_context {
    struct llama_context           * ctx;
    struct llama_context           * draft_ctx;
    argus_model                    * model_ref;
    argus_model                    * draft_model_ref;
    int32_t                          spec_draft_n_max;
    bool                             enable_draft_mtp;
    struct llama_context           * vocoder_ctx;
    const argus_model              * vocoder_model_ref;
    std::mutex                       mtx;                 // per-context concurrency guard

    // Sequence-isolated persistent sampler state slots
    std::vector<argus_seq_sampler>   seq_samplers;
    int32_t                          last_decoded_seq_id = -1;
};

// Internal helper to discard a sequence slot's pending sample and rebuild its chain
extern "C" bool discard_slot_pending_preserving_rng(argus_context_t * ctx, int32_t seq_id);

#endif // ARGUS_INTERNAL_H
