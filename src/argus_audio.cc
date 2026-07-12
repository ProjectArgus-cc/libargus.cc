/**
 * @file argus_audio.cc
 * @brief Audio transcription engine powered by Whisper transformer blocks.
 *
 * Implements the unmanaged execution pathways mapped to the public audio handling
 * boundaries of libargus.h, maintaining a zero-copy pipeline for Project Panama.
 */

#include "libargus.h"
#include "whisper.h"

#include <cstring>
#include <mutex>

// Internal structural context footprint wrapping the native Whisper layout
struct argus_audio_context {
    struct whisper_context * ctx;
    int32_t                 cpu_threads;
    std::mutex              mtx;   // per-context concurrency guard
};

extern "C" {

argus_audio_context_t * argus_audio_init(const argus_audio_params_t * params) {
    if (!params || !params->whisper_model_path) {
        return nullptr;
    }

    // Configure Whisper context params for GPU offloading
    struct whisper_context_params wctx_params = whisper_context_default_params();
    wctx_params.use_gpu    = (params->gpu_layers > 0);
    wctx_params.gpu_device = 0;

    struct whisper_context * wctx = whisper_init_from_file_with_params(params->whisper_model_path, wctx_params);
    if (!wctx) {
        return nullptr;
    }

    argus_audio_context_t * argus_ctx = new argus_audio_context();
    argus_ctx->ctx         = wctx;
    argus_ctx->cpu_threads = (params->cpu_threads > 0) ? params->cpu_threads : 4;
    return argus_ctx;
}

void argus_audio_free(argus_audio_context_t * ctx) {
    if (ctx) {
        if (ctx->ctx) {
            whisper_free(ctx->ctx);
        }
        delete ctx;
    }
}

int32_t argus_transcribe_audio(argus_audio_context_t * ctx, const float * pcm_data, int32_t sample_count, char * out_text, int32_t max_chars) {
    if (!ctx || !pcm_data || sample_count <= 0 || !out_text || max_chars <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    // Configure standard greedy decoding parameters for low-latency acoustic execution
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Suppress internal framework console outputs to maximize stream isolation
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;

    // Apply the configured CPU thread count for acoustic matrix math
    wparams.n_threads = ctx->cpu_threads;

    // Evaluate the raw float array through the acoustic encoder/decoder passes
    int run_status = whisper_full(ctx->ctx, wparams, pcm_data, sample_count);
    if (run_status != 0) {
        return -2;
    }

    // Safely parse segments from the completed compute graph into the destination pointer
    int n_segments = whisper_full_n_segments(ctx->ctx);
    int32_t total_written_chars = 0;
    out_text[0] = '\0';

    for (int i = 0; i < n_segments; ++i) {
        const char * segment_text = whisper_full_get_segment_text(ctx->ctx, i);
        if (segment_text) {
            size_t segment_len = strlen(segment_text);

            // Check absolute buffer boundary ceilings before running memory mutations
            if (total_written_chars + segment_len + 1 < (size_t)max_chars) {
                strcat(out_text, segment_text);
                total_written_chars += segment_len;
            } else {
                // Execute clean truncation right up to the maximum character allocation cap
                size_t remaining_space = max_chars - total_written_chars - 1;
                strncat(out_text, segment_text, remaining_space);
                total_written_chars += remaining_space;
                break;
            }
        }
    }

    return total_written_chars;
}

} // extern "C"