#include "libargus.h"
#include "argus_internal.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cstring>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include "argus_execution_lock.h"
#include <limits>
#include <cstdio>
#ifdef ARGUS_TESTING
#include "native_test_hooks.h"
#endif

// Internal wrappers
struct argus_multimodal {
    std::atomic<uint32_t> refs{1};
    mtmd_context * ctx = nullptr;
    std::mutex execution_mutex;
#ifdef ARGUS_TESTING
    bool test_evaluator = false;
#endif
    argus_model_t * model_ref = nullptr;
};

// Deleted struct argus_bitmap wrapper to eliminate native allocation loop

struct argus_video {
    mtmd_helper_video * video;
    argus_multimodal_t * mctx;
    std::mutex mtx;
#ifdef ARGUS_TESTING
    int test_remaining = -1;
    int test_next = 0;
#endif
};

struct argus_input_chunks {
    mtmd_input_chunks * chunks;
};

#ifdef ARGUS_TESTING
static std::atomic<int> test_eval_result{0};
#endif
extern "C" {
#ifdef ARGUS_TESTING
void argus_test_set_eval_result(int result) { test_eval_result.store(result); }
argus_multimodal_t * argus_test_projector(argus_model_t * model) {
    auto wrapper = std::make_unique<argus_multimodal>();
    if (!argus_model_retain(model)) return nullptr;
    wrapper->model_ref = model;
    wrapper->test_evaluator = true;
    return wrapper.release();
}
argus_input_chunks_t * argus_test_chunks(void) {
    auto wrapper = std::make_unique<argus_input_chunks>();
    wrapper->chunks = mtmd_test_create_input_chunks();
    return wrapper.release();
}
argus_video_t * argus_test_video(argus_multimodal_t * mctx, int count) {
    auto wrapper = std::make_unique<argus_video>();
    argus_multimodal_retain(mctx);
    wrapper->mctx = mctx;
    wrapper->video = nullptr;
    wrapper->test_remaining = count;
    return wrapper.release();
}
#endif


argus_multimodal_t * argus_multimodal_init(const argus_model_t * model, const argus_multimodal_params_t * params) {
    try {
        clear_last_error();
        if (!model || !params || !params->mmproj_path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid model or multimodal parameters");
            return nullptr;
        }

        auto * non_const_model = const_cast<argus_model_t *>(model);
        if (!argus_model_retain(non_const_model)) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "failed to retain base model for multimodal context");
            return nullptr;
        }

        std::unique_ptr<argus_model_t, decltype(&argus_model_release)> owned_model(
            non_const_model, &argus_model_release);
#ifdef ARGUS_TESTING
        argus_test_allocation_checkpoint();
#endif
        auto mparams = mtmd_context_params_default();
        mparams.use_gpu = params->use_gpu;
        mparams.n_threads = params->cpu_threads > 0 ? params->cpu_threads : 4;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> owned_context(
            mtmd_init_from_file(params->mmproj_path, model->model, mparams), &mtmd_free);
        if (!owned_context) {
            set_last_error(ARGUS_ERROR_MODEL_LOAD, "failed to initialize multimodal context from file");
            return nullptr;
        }
        auto wrapper = std::make_unique<argus_multimodal>();
        wrapper->ctx = owned_context.release();
        wrapper->model_ref = owned_model.release();
        return wrapper.release();
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during multimodal context init");
    }
    return nullptr;
}

bool argus_multimodal_retain(argus_multimodal_t * mctx) {
    if (!mctx) {
        return false;
    }
    mctx->refs.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void argus_multimodal_release(argus_multimodal_t * mctx) {
    if (!mctx) {
        return;
    }
    if (mctx->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        try {
            clear_last_error();
            if (mctx->ctx) {
                mtmd_free(mctx->ctx);
                mctx->ctx = nullptr;
            }
            if (mctx->model_ref) {
                argus_model_release(mctx->model_ref);
                mctx->model_ref = nullptr;
            }
#ifdef ARGUS_TESTING
            argus_test_notify(6, mctx);
#endif
            delete mctx;
        } catch (...) {
            // Suppress exceptions during destruction
        }
    }
}

void argus_multimodal_free(argus_multimodal_t * mctx) {
    argus_multimodal_release(mctx);
}

bool argus_multimodal_support_vision(const argus_multimodal_t * mctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_multimodal_support_vision", [&]() -> bool {
        return mctx && mctx->ctx ? mtmd_support_vision(mctx->ctx) : false;
    });
}

bool argus_multimodal_support_audio(const argus_multimodal_t * mctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_multimodal_support_audio", [&]() -> bool {
        return mctx && mctx->ctx ? mtmd_support_audio(mctx->ctx) : false;
    });
}

bool argus_multimodal_support_video(const argus_multimodal_t * mctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, false, "argus_multimodal_support_video", [&]() -> bool {
        return mctx && mctx->ctx ? mtmd_helper_support_video(mctx->ctx) : false;
    });
}

int32_t argus_multimodal_get_audio_sample_rate(const argus_multimodal_t * mctx) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -1, "argus_multimodal_get_audio_sample_rate", [&]() -> int32_t {
        return mctx && mctx->ctx ? (int32_t)mtmd_get_audio_sample_rate(mctx->ctx) : -1;
    });
}

argus_bitmap_t * argus_bitmap_from_rgb(uint32_t width, uint32_t height, const uint8_t * rgb_data) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_bitmap_t *)nullptr, "argus_bitmap_from_rgb", [&]() -> argus_bitmap_t * {
        if (!rgb_data || width == 0 || height == 0 ||
            width > static_cast<uint32_t>(INT32_MAX) || height > static_cast<uint32_t>(INT32_MAX) ||
            static_cast<uint64_t>(width) * height > static_cast<uint64_t>(INT32_MAX) / 3) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "RGB dimensions must be positive and fit a signed 32-bit byte extent");
            return nullptr;
        }
        return reinterpret_cast<argus_bitmap_t*>(mtmd_bitmap_init(width, height, rgb_data));
    });
}

argus_bitmap_t * argus_bitmap_from_pcm(const float * pcm_data, int32_t n_samples) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_bitmap_t *)nullptr, "argus_bitmap_from_pcm", [&]() -> argus_bitmap_t * {
        if (!pcm_data || n_samples <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid pcm_data or n_samples");
            return nullptr;
        }
        return reinterpret_cast<argus_bitmap_t*>(mtmd_bitmap_init_from_audio((size_t)n_samples, pcm_data));
    });
}

argus_bitmap_t * argus_bitmap_load_file(argus_multimodal_t * mctx, const char * path, bool placeholder) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_bitmap_t *)nullptr, "argus_bitmap_load_file", [&]() -> argus_bitmap_t * {
        if (!mctx || !mctx->ctx || !path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid mctx or file path");
            return nullptr;
        }

        struct mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(mctx->ctx, path, placeholder);
        if (wrapper.video_ctx) {
            mtmd_helper_video_free(wrapper.video_ctx);
            if (wrapper.bitmap) {
                mtmd_bitmap_free(wrapper.bitmap);
            }
            return nullptr;
        }

        return reinterpret_cast<argus_bitmap_t*>(wrapper.bitmap);
    });
}

argus_bitmap_t * argus_bitmap_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, bool placeholder) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_bitmap_t *)nullptr, "argus_bitmap_load_buffer", [&]() -> argus_bitmap_t * {
        if (!mctx || !mctx->ctx || !buffer || size <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid mctx or buffer parameters");
            return nullptr;
        }

        struct mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(mctx->ctx, buffer, (size_t)size, placeholder);
        if (wrapper.video_ctx) {
            mtmd_helper_video_free(wrapper.video_ctx);
            if (wrapper.bitmap) {
                mtmd_bitmap_free(wrapper.bitmap);
            }
            return nullptr;
        }

        return reinterpret_cast<argus_bitmap_t*>(wrapper.bitmap);
    });
}

void argus_bitmap_free(argus_bitmap_t * bitmap) {
    argus_guard_void("argus_bitmap_free", [&]() {
        if (bitmap) {
            mtmd_bitmap_free(reinterpret_cast<mtmd_bitmap*>(bitmap));
        }
    });
}

argus_video_t * argus_video_load_file(argus_multimodal_t * mctx, const char * path, float fps_target, int64_t timestamp_interval_ms) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_video_t *)nullptr, "argus_video_load_file", [&]() -> argus_video_t * {
        if (!mctx || !mctx->ctx || !path) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid mctx or file path");
            return nullptr;
        }

        struct mtmd_helper_video_init_params params = mtmd_helper_video_init_params_default();
        params.fps_target = fps_target;
        params.timestamp_interval_ms = timestamp_interval_ms;

        std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)> owned_video(
            mtmd_helper_video_init(mctx->ctx, path, params),
            &mtmd_helper_video_free
        );
        if (!owned_video) {
            return nullptr;
        }

        auto wrapper = std::make_unique<argus_video>();
        argus_multimodal_retain(mctx);
        wrapper->mctx = mctx;
        wrapper->video = owned_video.release();
        return wrapper.release();
    });
}

argus_video_t * argus_video_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, float fps_target, int64_t timestamp_interval_ms) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_video_t *)nullptr, "argus_video_load_buffer", [&]() -> argus_video_t * {
        if (!mctx || !mctx->ctx || !buffer || size <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid mctx or buffer");
            return nullptr;
        }

        struct mtmd_helper_video_init_params params = mtmd_helper_video_init_params_default();
        params.fps_target = fps_target;
        params.timestamp_interval_ms = timestamp_interval_ms;

        std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)> owned_video(
            mtmd_helper_video_init_from_buf(mctx->ctx, buffer, (size_t)size, params),
            &mtmd_helper_video_free
        );
        if (!owned_video) {
            return nullptr;
        }

        auto wrapper = std::make_unique<argus_video>();
        argus_multimodal_retain(mctx);
        wrapper->mctx = mctx;
        wrapper->video = owned_video.release();
        return wrapper.release();
    });
}


void argus_video_free(argus_video_t * video) {
    argus_guard_void("argus_video_free", [&]() {
        if (video) {
            {
                std::lock_guard<std::mutex> lock(video->mtx);
                if (video->video) {
                    mtmd_helper_video_free(video->video);
                    video->video = nullptr;
                }
            }
            if (video->mctx) {
                argus_multimodal_release(video->mctx);
                video->mctx = nullptr;
            }
            delete video;
        }
    });
}

int32_t argus_video_read_next(argus_video_t * video, argus_bitmap_t ** out_bitmap, char * out_text, int32_t max_chars) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -2, "argus_video_read_next", [&]() -> int32_t {
        if (!video || !out_bitmap || !out_text || max_chars <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid video arguments");
            return -2;
        }

        auto lock = argus_execution_lock(video->mtx, video, 5);
#ifdef ARGUS_TESTING
        if (video->test_remaining >= 0) {
            argus_test_notify(5, video);
            *out_bitmap = nullptr;
            out_text[0] = '\0';
            if (video->test_remaining == 0) return -1;
            std::snprintf(out_text, max_chars, "%d", video->test_next++);
            --video->test_remaining;
            return 0;
        }
#endif
        if (!video->video) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "video iterator already closed or invalid");
            return -2;
        }

        mtmd_bitmap * mtmd_b = nullptr;
        char * mtmd_t = nullptr;

        int32_t res = mtmd_helper_video_read_next(video->video, &mtmd_b, &mtmd_t);
        if (res == 0) {
            if (mtmd_b) {
                *out_bitmap = reinterpret_cast<argus_bitmap_t*>(mtmd_b);
                out_text[0] = '\0';
            } else if (mtmd_t) {
                *out_bitmap = nullptr;
                strncpy(out_text, mtmd_t, max_chars - 1);
                out_text[max_chars - 1] = '\0';
                free(mtmd_t);
            } else {
                *out_bitmap = nullptr;
                out_text[0] = '\0';
            }
        } else {
            *out_bitmap = nullptr;
            out_text[0] = '\0';
        }
        return res;
    });
}

argus_input_chunks_t * argus_input_chunks_init(void) {
    return argus_guard(ARGUS_ERROR_INTERNAL, (argus_input_chunks_t *)nullptr, "argus_input_chunks_init", [&]() -> argus_input_chunks_t * {
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> owned_chunks(
            mtmd_input_chunks_init(),
            &mtmd_input_chunks_free
        );
        if (!owned_chunks) {
            return nullptr;
        }

        auto wrapper = std::make_unique<argus_input_chunks>();
        wrapper->chunks = owned_chunks.release();
        return wrapper.release();
    });
}


void argus_input_chunks_free(argus_input_chunks_t * chunks) {
    argus_guard_void("argus_input_chunks_free", [&]() {
        if (chunks) {
            if (chunks->chunks) {
                mtmd_input_chunks_free(chunks->chunks);
            }
            delete chunks;
        }
    });
}

int32_t argus_multimodal_tokenize_n(
    argus_multimodal_t * mctx,
    argus_input_chunks_t * output,
    const char * text,
    size_t text_len,
    bool add_bos,
    const argus_bitmap_t ** bitmaps,
    int32_t n_bitmaps) {
    try {
        clear_last_error();
        if (!mctx || !output || !text || n_bitmaps < 0 || (n_bitmaps > 0 && !bitmaps)) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid multimodal tokenize arguments");
            return -1;
        }

        mtmd_input_text input_text;
        input_text.text = text;
        input_text.text_len = (size_t)text_len;
        input_text.add_special = add_bos;
        input_text.parse_special = true;

        std::vector<const mtmd_bitmap *> mtmd_bitmaps(n_bitmaps);
        for (int32_t i = 0; i < n_bitmaps; ++i) {
            if (!bitmaps[i]) {
                set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null bitmap pointer in array");
                return -1;
            }
            mtmd_bitmaps[i] = reinterpret_cast<const mtmd_bitmap*>(bitmaps[i]);
        }

        return mtmd_tokenize(mctx->ctx, output->chunks, &input_text, mtmd_bitmaps.data(), (size_t)n_bitmaps);
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
        return -1;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
        return -1;
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during multimodal tokenize");
        return -1;
    }
}

int32_t argus_multimodal_tokenize(
    argus_multimodal_t * mctx,
    argus_input_chunks_t * output,
    const char * text,
    bool add_bos,
    const argus_bitmap_t ** bitmaps,
    int32_t n_bitmaps) {
    if (!text) {
        return -1;
    }
    return argus_multimodal_tokenize_n(mctx, output, text, strlen(text), add_bos, bitmaps, n_bitmaps);
}

int32_t argus_eval_multimodal_chunks(
    argus_multimodal_t * mctx,
    argus_context_t * ctx,
    const argus_input_chunks_t * chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t * out_new_n_past) {
    try {
        clear_last_error();
        if (!mctx || !ctx || !chunks || !out_new_n_past) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid eval multimodal arguments");
            return -1;
        }

        if (seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size() ||
            n_past < 0 || n_batch <= 0 || n_batch > static_cast<int64_t>(llama_n_batch(ctx->ctx)) ||
            mctx->model_ref != ctx->model_ref) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid sequence, position, batch size, or projector model association");
            return -1;
        }
        int64_t end_position = n_past;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks->chunks); ++i) {
            const auto positions = mtmd_input_chunk_get_n_pos(mtmd_input_chunks_get(chunks->chunks, i));
            if (positions < 0 || positions > std::numeric_limits<int32_t>::max() - end_position) {
                set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "multimodal positions exceed the signed 32-bit range");
                return -1;
            }
            end_position += positions;
        }

        // Lock order: projector, then text context. Hold through embedding consumption.
        auto projector_lock = argus_execution_lock(mctx->execution_mutex, mctx, 1);
        auto context_lock = argus_execution_lock(ctx->mtx, ctx, 2);
        if (mtmd_input_chunks_size(chunks->chunks) == 0) {
            *out_new_n_past = n_past;
            return 0;
        }
        // The pinned helper only requests logits on a terminal TEXT chunk.
        // Media-only evaluation must not advertise a nonexistent output buffer.
        const auto * terminal = mtmd_input_chunks_get(chunks->chunks, mtmd_input_chunks_size(chunks->chunks) - 1);
        size_t terminal_tokens = 0;
        if (mtmd_input_chunk_get_type(terminal) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            mtmd_input_chunk_get_tokens_text(terminal, &terminal_tokens);
        }
        const bool produces_logits = logits_last && terminal_tokens > 0;

        // Reconcile and discard any pending text sample prior to multimodal evaluation
        discard_slot_pending_preserving_rng(ctx, seq_id);

        // Invalidate pending logits before starting multimodal projection evaluation
        ctx->seq_samplers[seq_id].has_logits = false;
        ctx->seq_samplers[seq_id].last_logits_pos = -1;
        // llama owns one logits buffer across all sequences; failure invalidates its owner.
        ctx->last_decoded_seq_id = -1;

        llama_pos new_n_past_val = n_past;

#ifdef ARGUS_TESTING
        argus_test_notify(3, mctx);
#endif
        int32_t res;
#ifdef ARGUS_TESTING
        if (mctx->test_evaluator) {
            new_n_past_val = n_past + 1;
            res = test_eval_result.load();
        } else
#endif
        res = mtmd_helper_eval_chunks(
            mctx->ctx,
            ctx->ctx,
            chunks->chunks,
            n_past,
            seq_id,
            n_batch,
            logits_last,
            &new_n_past_val
        );

        if (res == 0) {
            *out_new_n_past = (int32_t)new_n_past_val;
            ctx->last_decoded_seq_id = seq_id;
            ctx->seq_samplers[seq_id].has_logits = produces_logits;
            ctx->seq_samplers[seq_id].last_logits_pos = produces_logits ? ((int32_t)new_n_past_val - 1) : -1;
        }

        return res;
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
        return -1;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
        return -1;
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during multimodal chunk eval");
        return -1;
    }
}
} // extern "C"
