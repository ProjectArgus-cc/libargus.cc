#ifndef ARGUS_INTERNAL_H
#define ARGUS_INTERNAL_H

#include "libargus.h"
#include "llama.h"
#include <mutex>

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
    std::mutex             mtx;        // per-context concurrency guard
};

#endif // ARGUS_INTERNAL_H
