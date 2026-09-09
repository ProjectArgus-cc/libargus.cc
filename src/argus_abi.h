#pragma once
#include <array>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include "libargus.h"
#include "argus_build_info.h"

// Derive the wire layout from the compiler, including every non-padding field.
inline const auto & argus_diagnostic_info() {
    struct Info { std::array<char, 4096> data{}; int length = 0; };
    static const Info info = [] {
        Info result;
        auto append = [&](const char * format, auto... values) {
            if (result.length < 0) return;
            const size_t remaining = result.data.size() - result.length;
            const int size = std::snprintf(result.data.data() + result.length, remaining, format, values...);
            if (size < 0 || static_cast<size_t>(size) >= remaining) result.length = -1;
            else result.length += size;
        };
        append("%sabi_layouts=", ARGUS_BUILD_INFO);
        append("%zu:%zu:%zu,%zu,%zu", sizeof(argus_model_params_t), alignof(argus_model_params_t), offsetof(argus_model_params_t, model_path), offsetof(argus_model_params_t, gpu_layers), offsetof(argus_model_params_t, use_mlock));
        append(";%zu:%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu", sizeof(argus_context_params_t), alignof(argus_context_params_t), offsetof(argus_context_params_t, draft_model), offsetof(argus_context_params_t, context_length), offsetof(argus_context_params_t, cpu_threads), offsetof(argus_context_params_t, type_k), offsetof(argus_context_params_t, type_v), offsetof(argus_context_params_t, spec_draft_n_max), offsetof(argus_context_params_t, u_batch), offsetof(argus_context_params_t, n_seq_max), offsetof(argus_context_params_t, enable_draft_mtp), offsetof(argus_context_params_t, embeddings), offsetof(argus_context_params_t, kv_unified));
        append(";%zu:%zu:%zu,%zu,%zu", sizeof(argus_audio_params_t), alignof(argus_audio_params_t), offsetof(argus_audio_params_t, whisper_model_path), offsetof(argus_audio_params_t, cpu_threads), offsetof(argus_audio_params_t, gpu_layers));
        append(";%zu:%zu:%zu,%zu,%zu,%zu,%zu,%zu", sizeof(argus_token_batch_t), alignof(argus_token_batch_t), offsetof(argus_token_batch_t, tokens), offsetof(argus_token_batch_t, n_tokens), offsetof(argus_token_batch_t, start_pos), offsetof(argus_token_batch_t, seq_id), offsetof(argus_token_batch_t, request_logits), offsetof(argus_token_batch_t, abort_flag));
        append(";%zu:%zu:%zu,%zu,%zu", sizeof(argus_multimodal_params_t), alignof(argus_multimodal_params_t), offsetof(argus_multimodal_params_t, mmproj_path), offsetof(argus_multimodal_params_t, cpu_threads), offsetof(argus_multimodal_params_t, use_gpu));
        append(";%zu:%zu:%zu,%zu", sizeof(argus_logit_bias_t), alignof(argus_logit_bias_t), offsetof(argus_logit_bias_t, token), offsetof(argus_logit_bias_t, bias));
        append(";%zu:%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu", sizeof(argus_sampler_params_t), alignof(argus_sampler_params_t), offsetof(argus_sampler_params_t, temperature), offsetof(argus_sampler_params_t, repeat_penalty), offsetof(argus_sampler_params_t, repeat_last_n), offsetof(argus_sampler_params_t, frequency_penalty), offsetof(argus_sampler_params_t, presence_penalty), offsetof(argus_sampler_params_t, top_p), offsetof(argus_sampler_params_t, min_p), offsetof(argus_sampler_params_t, top_k), offsetof(argus_sampler_params_t, dry_multiplier), offsetof(argus_sampler_params_t, dry_base), offsetof(argus_sampler_params_t, dry_allowed_length), offsetof(argus_sampler_params_t, dry_penalty_last_n), offsetof(argus_sampler_params_t, seed));
        append("\n");
        return result;
    }();
    return info;
}
