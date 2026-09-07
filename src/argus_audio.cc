/**
 * @file argus_audio.cc
 * @brief Audio transcription engine powered by Whisper transformer blocks.
 *
 * Implements the unmanaged execution pathways mapped to the public audio handling
 * boundaries of libargus.h, maintaining a zero-copy pipeline for Project Panama.
 */

#include "libargus.h"
#include "argus_internal.h"
#include "whisper.h"

#include <cstring>
#include <mutex>
#include <new>

// Internal structural context footprint wrapping the native Whisper layout
struct argus_audio_context {
    struct whisper_context * ctx;
    int32_t                 cpu_threads;
    std::mutex              mtx;   // per-context concurrency guard
};

extern "C" {

argus_audio_context_t * argus_audio_init(const argus_audio_params_t * params) {
    try {
        clear_last_error();
        if (!params || !params->whisper_model_path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "audio params or model path is NULL");
            return nullptr;
        }

        // Configure Whisper context params for GPU offloading
        struct whisper_context_params wctx_params = whisper_context_default_params();
        wctx_params.use_gpu    = (params->gpu_layers > 0);
        wctx_params.gpu_device = 0;

        struct whisper_context * wctx = whisper_init_from_file_with_params(params->whisper_model_path, wctx_params);
        if (!wctx) {
            set_last_error(ARGUS_ERROR_MODEL_LOAD, "failed to load whisper model from file");
            return nullptr;
        }

        argus_audio_context_t * argus_ctx = nullptr;
        try {
            argus_ctx = new argus_audio_context();
            argus_ctx->ctx         = wctx;
            argus_ctx->cpu_threads = (params->cpu_threads > 0) ? params->cpu_threads : 4;
            argus_backend_resource_inc();
            return argus_ctx;
        } catch (...) {
            whisper_free(wctx);
            throw;
        }
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during audio context init");
    }
    return nullptr;
}

void argus_audio_free(argus_audio_context_t * ctx) {
    if (!ctx) {
        return;
    }
    try {
        clear_last_error();
        if (ctx->ctx) {
            whisper_free(ctx->ctx);
            ctx->ctx = nullptr;
        }
        delete ctx;
        argus_backend_resource_dec();
    } catch (...) {
        // Suppress exceptions during destruction
    }
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
            set_last_error(ARGUS_ERROR_BACKEND, "whisper transcription failed");
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
                    // Zero-copy linear copying using memcpy instead of quadratic strcat
                    std::memcpy(out_text + total_written_chars, segment_text, segment_len);
                    total_written_chars += segment_len;
                    out_text[total_written_chars] = '\0';
                } else {
                    // Execute clean truncation right up to the maximum character allocation cap
                    size_t remaining_space = max_chars - total_written_chars - 1;
                    if (remaining_space > 0) {
                        std::memcpy(out_text + total_written_chars, segment_text, remaining_space);
                        total_written_chars += remaining_space;
                        out_text[total_written_chars] = '\0';

                        // Strict UTF-8 continuation byte boundary enforcement:
                        // If the last character was cut off (continuation byte high two bits are '10xxxxxx'),
                        // backtrack to the starting leading byte of the multi-byte char and terminate.
                        int32_t idx = total_written_chars - 1;
                        while (idx >= 0 && (out_text[idx] & 0xC0) == 0x80) {
                            idx--;
                        }
                        if (idx >= 0 && (out_text[idx] & 0x80) != 0) {
                            out_text[idx] = '\0';
                            total_written_chars = idx;
                        }
                    }
                    break;
                }
            }
        }

        return total_written_chars;
    });
}

} // extern "C"