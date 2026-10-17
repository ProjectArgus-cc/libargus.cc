/**
 * @file argus_audio.cc
 * @brief Audio transcription engine powered by Whisper transformer blocks.
 *
 * Implements the unmanaged execution pathways mapped to the public audio handling
 * boundaries of libargus.h, maintaining a zero-copy pipeline for Project Panama.
 */

#include "libargus.h"
#include "argus_internal.h"
#include "argus_execution_lock.h"
#include "whisper.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#ifdef ARGUS_TESTING
#include "native_test_hooks.h"
#endif

// Internal structural context footprint wrapping the native Whisper layout
struct argus_audio_context {
    struct whisper_context * ctx = nullptr;
    int32_t                 cpu_threads = 4;
    std::mutex              mtx;   // per-context concurrency guard
#ifdef ARGUS_TESTING
    std::string             test_transcript;
    bool                    is_test = false;
#endif
};

extern "C" {

#ifdef ARGUS_TESTING
argus_audio_context_t * argus_test_audio(const char * transcript) {
    auto wrapper = std::make_unique<argus_audio_context>();
    wrapper->ctx = nullptr;
    wrapper->cpu_threads = 2;
    wrapper->test_transcript = transcript ? transcript : "";
    wrapper->is_test = true;
    argus_backend_resource_inc();
    return wrapper.release();
}
#endif

argus_audio_context_t * argus_audio_init(const argus_audio_params_t * params) {
    return argus_guard(ARGUS_ERROR_INTERNAL, static_cast<argus_audio_context_t *>(nullptr), "argus_audio_init", [&]() -> argus_audio_context_t * {
        if (!params || !params->whisper_model_path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "audio params or model path is NULL");
            return nullptr;
        }

        // Configure Whisper context params for GPU offloading
        struct whisper_context_params wctx_params = whisper_context_default_params();
        wctx_params.use_gpu    = (params->gpu_layers > 0);
        wctx_params.gpu_device = 0;

        std::unique_ptr<struct whisper_context, decltype(&whisper_free)> wctx(
            whisper_init_from_file_with_params(params->whisper_model_path, wctx_params),
            &whisper_free);
        if (!wctx) {
            set_last_error(ARGUS_ERROR_MODEL_LOAD, "failed to load whisper model from file");
            return nullptr;
        }

        auto argus_ctx = std::make_unique<argus_audio_context>();
        argus_ctx->ctx         = wctx.release();
        argus_ctx->cpu_threads = (params->cpu_threads > 0) ? params->cpu_threads : 4;
        argus_backend_resource_inc();
        return argus_ctx.release();
    });
}

void argus_audio_free(argus_audio_context_t * ctx) {
    if (!ctx) {
        return;
    }
    argus_guard_void("argus_audio_free", [&]() {
        if (ctx->ctx) {
            whisper_free(ctx->ctx);
            ctx->ctx = nullptr;
        }
        delete ctx;
        argus_backend_resource_dec();
    });
}

void argus_audio_set_n_threads(argus_audio_context_t * ctx, int32_t n_threads) {
    argus_guard_void("argus_audio_set_n_threads", [&]() {
        if (!ctx) {
            return;
        }
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->cpu_threads = (n_threads > 0) ? n_threads : 1;
    });
}

int32_t argus_audio_get_n_threads(argus_audio_context_t * ctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_audio_get_n_threads", [&]() -> int32_t {
        if (!ctx) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(ctx->mtx);
        return ctx->cpu_threads;
    });
}

int32_t argus_transcribe_audio(argus_audio_context_t * ctx, const float * pcm_data, int32_t sample_count, char * out_text, int32_t max_chars) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_transcribe_audio", [&]() -> int32_t {
        if (!ctx || !pcm_data || sample_count <= 0 || !out_text || max_chars <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid transcribe parameters");
            return -1;
        }

        auto lock = argus_execution_lock(ctx->mtx, ctx, 7);
#ifdef ARGUS_TESTING
        argus_test_notify(7, ctx);
        const bool is_test = ctx->is_test;
#else
        const bool is_test = false;
#endif
        int n_segments = 1;
        if (!is_test) {
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
                set_last_error(ARGUS_ERROR_BACKEND, "whisper transcription failed");
                return -2;
            }

            // Safely parse segments from the completed compute graph into the destination pointer
            n_segments = whisper_full_n_segments(ctx->ctx);
        }

        int32_t total_written_chars = 0;
        out_text[0] = '\0';

        for (int i = 0; i < n_segments; ++i) {
#ifdef ARGUS_TESTING
            const char * segment_text = is_test ? ctx->test_transcript.c_str() : whisper_full_get_segment_text(ctx->ctx, i);
#else
            const char * segment_text = whisper_full_get_segment_text(ctx->ctx, i);
#endif
            if (segment_text) {
                size_t segment_len = strlen(segment_text);

                // Check absolute buffer boundary ceilings before running memory mutations
                if ((size_t)total_written_chars + segment_len + 1 <= (size_t)max_chars) {
                    // Zero-copy linear copying using memcpy instead of quadratic strcat
                    std::memcpy(out_text + total_written_chars, segment_text, segment_len);
                    total_written_chars += (int32_t)segment_len;
                    out_text[total_written_chars] = '\0';
                } else {
                    // Execute clean truncation right up to the maximum character allocation cap
                    size_t remaining_space = (size_t)max_chars - (size_t)total_written_chars - 1;
                    if (remaining_space > 0) {
                        std::memcpy(out_text + total_written_chars, segment_text, remaining_space);
                        total_written_chars += (int32_t)remaining_space;
                        out_text[total_written_chars] = '\0';
                        total_written_chars = argus_utf8_truncate_incomplete_tail(out_text, total_written_chars);
                    }
                    break;
                }
            }
        }

        return total_written_chars;
    });
}

} // extern "C"