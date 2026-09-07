#include "libargus.h"
#include "argus_internal.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cstring>
#include <mutex>
#include <thread>
#include <chrono>

// Internal wrappers
struct argus_multimodal {
    mtmd_context * ctx;
    argus_model_t * model_ref;
};

// Deleted struct argus_bitmap wrapper to eliminate native allocation loop

struct argus_video {
    mtmd_helper_video * video;
};

struct argus_input_chunks {
    mtmd_input_chunks * chunks;
};

extern "C" {

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

        struct mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params->use_gpu;
        mparams.n_threads = (params->cpu_threads > 0) ? params->cpu_threads : 4;

        mtmd_context * mctx = mtmd_init_from_file(params->mmproj_path, model->model, mparams);
        if (!mctx) {
            argus_model_release(non_const_model);
            set_last_error(ARGUS_ERROR_MODEL_LOAD, "failed to initialize multimodal context from file");
            return nullptr;
        }

        argus_multimodal_t * argus_mctx = nullptr;
        try {
            argus_mctx = new argus_multimodal();
            argus_mctx->ctx = mctx;
            argus_mctx->model_ref = non_const_model;
            return argus_mctx;
        } catch (...) {
            mtmd_free(mctx);
            argus_model_release(non_const_model);
            throw;
        }
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during multimodal context init");
    }
    return nullptr;
}

void argus_multimodal_free(argus_multimodal_t * mctx) {
    if (!mctx) {
        return;
    }
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
        delete mctx;
    } catch (...) {
        // Suppress exceptions during destruction
    }
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
        if (!rgb_data) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "rgb_data is NULL");
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

        mtmd_helper_video * video = mtmd_helper_video_init(mctx->ctx, path, params);
        if (!video) {
            return nullptr;
        }

        argus_video_t * v = new argus_video();
        v->video = video;
        return v;
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

        mtmd_helper_video * video = mtmd_helper_video_init_from_buf(mctx->ctx, buffer, (size_t)size, params);
        if (!video) {
            return nullptr;
        }

        argus_video_t * v = new argus_video();
        v->video = video;
        return v;
    });
}

void argus_video_free(argus_video_t * video) {
    argus_guard_void("argus_video_free", [&]() {
        if (video) {
            if (video->video) {
                mtmd_helper_video_free(video->video);
            }
            delete video;
        }
    });
}

int32_t argus_video_read_next(argus_video_t * video, argus_bitmap_t ** out_bitmap, char * out_text, int32_t max_chars) {
    return argus_guard(ARGUS_ERROR_INTERNAL, -2, "argus_video_read_next", [&]() -> int32_t {
        if (!video || !video->video || !out_bitmap || !out_text || max_chars <= 0) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid video arguments");
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
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (!chunks) {
            return nullptr;
        }

        argus_input_chunks_t * argus_chunks = new argus_input_chunks();
        argus_chunks->chunks = chunks;
        return argus_chunks;
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
        if (!mctx || !output || !text || (n_bitmaps > 0 && !bitmaps)) {
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
    return argus_multimodal_tokenize_n(mctx, output, text, (int32_t)strlen(text), add_bos, bitmaps, n_bitmaps);
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

        if (seq_id < 0 || seq_id >= (int32_t)ctx->seq_samplers.size()) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "sequence ID out of range");
            return -1;
        }

        std::lock_guard<std::mutex> lock(ctx->mtx);

        // Reconcile and discard any pending text sample prior to multimodal evaluation
        discard_slot_pending_preserving_rng(ctx, seq_id);

        // Invalidate pending logits before starting multimodal projection evaluation
        ctx->seq_samplers[seq_id].has_logits = false;
        ctx->seq_samplers[seq_id].last_logits_pos = -1;
        if (ctx->last_decoded_seq_id == seq_id) {
            ctx->last_decoded_seq_id = -1;
        }

        llama_pos new_n_past_val = n_past;

        int32_t res = mtmd_helper_eval_chunks(
            mctx->ctx,
            ctx->ctx,
            chunks->chunks,
            n_past,
            seq_id,
            n_batch,
            logits_last,
            &new_n_past_val
        );

        *out_new_n_past = (int32_t)new_n_past_val;

        if (res == 0) {
            ctx->last_decoded_seq_id = seq_id;
            ctx->seq_samplers[seq_id].has_logits = logits_last;
            ctx->seq_samplers[seq_id].last_logits_pos = logits_last ? ((int32_t)new_n_past_val - 1) : -1;
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

int32_t argus_multimodal_test_lock_sync(argus_context_t * ctx, int32_t seq_id, int32_t hold_us) {
    if (!ctx) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx->mtx);

    if (seq_id >= 0 && seq_id < (int32_t)ctx->seq_samplers.size()) {
        discard_slot_pending_preserving_rng(ctx, seq_id);
        ctx->seq_samplers[seq_id].has_logits = false;
        ctx->seq_samplers[seq_id].last_logits_pos = -1;
        if (ctx->last_decoded_seq_id == seq_id) {
            ctx->last_decoded_seq_id = -1;
        }
    }

    if (hold_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(hold_us));
    }

    return 0;
}

} // extern "C"
