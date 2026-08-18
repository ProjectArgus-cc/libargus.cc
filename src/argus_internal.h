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

struct argus_context {
    struct llama_context * ctx;
    struct llama_context * draft_ctx;
    argus_model          * model_ref;
    argus_model          * draft_model_ref;
    int32_t                spec_draft_n_max;
    bool                   enable_draft_mtp;
    struct llama_context * vocoder_ctx;
    const argus_model    * vocoder_model_ref;
    std::mutex             mtx;        // per-context concurrency guard

    // Zero-allocation persistent sampler chain cache
    struct llama_sampler        * cached_sampler_chain;
    argus_sampler_params_t        cached_sparams;
    std::vector<argus_logit_bias_t> cached_biases;
    bool                          has_cached_sampler;
};

#endif // ARGUS_INTERNAL_H
