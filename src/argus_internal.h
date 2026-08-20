#ifndef ARGUS_INTERNAL_H
#define ARGUS_INTERNAL_H

#include "libargus.h"
#include "llama.h"
#include <mutex>
#include <vector>

struct argus_model {
    struct llama_model           * model;
    const  struct llama_vocab    * vocab;
};

struct argus_seq_sampler {
    struct llama_sampler          * chain              = nullptr;
    argus_sampler_params_t          cached_sparams     = {};
    std::vector<argus_logit_bias_t> cached_biases;
    std::vector<llama_token>        history;           // Pre-reserved to context_length
    int32_t                         last_logits_pos    = -1;
    bool                            has_logits         = false;
    bool                            has_cached_chain   = false;
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

#endif // ARGUS_INTERNAL_H
