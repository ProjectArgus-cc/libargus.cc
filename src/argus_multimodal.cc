#include "libargus.h"
#include "argus_internal.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cstring>
#include <mutex>

// Internal wrappers
struct argus_multimodal {
    mtmd_context * ctx;
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
    if (!model || !params || !params->mmproj_path) {
        return nullptr;
    }

    struct mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params->use_gpu;
    mparams.n_threads = (params->cpu_threads > 0) ? params->cpu_threads : 4;

    mtmd_context * mctx = mtmd_init_from_file(params->mmproj_path, model->model, mparams);
    if (!mctx) {
        return nullptr;
    }

    argus_multimodal_t * argus_mctx = new argus_multimodal();
    argus_mctx->ctx = mctx;
    return argus_mctx;
}

void argus_multimodal_free(argus_multimodal_t * mctx) {
    if (mctx) {
        if (mctx->ctx) {
            mtmd_free(mctx->ctx);
        }
        delete mctx;
    }
}

bool argus_multimodal_support_vision(const argus_multimodal_t * mctx) {
    return mctx ? mtmd_support_vision(mctx->ctx) : false;
}

bool argus_multimodal_support_audio(const argus_multimodal_t * mctx) {
    return mctx ? mtmd_support_audio(mctx->ctx) : false;
}

bool argus_multimodal_support_video(const argus_multimodal_t * mctx) {
    return mctx ? mtmd_helper_support_video(mctx->ctx) : false;
}

int32_t argus_multimodal_get_audio_sample_rate(const argus_multimodal_t * mctx) {
    return mctx ? (int32_t)mtmd_get_audio_sample_rate(mctx->ctx) : -1;
}

argus_bitmap_t * argus_bitmap_from_rgb(uint32_t width, uint32_t height, const uint8_t * rgb_data) {
    if (!rgb_data) {
        return nullptr;
    }

    return reinterpret_cast<argus_bitmap_t*>(mtmd_bitmap_init(width, height, rgb_data));
}

argus_bitmap_t * argus_bitmap_from_pcm(const float * pcm_data, int32_t n_samples) {
    if (!pcm_data || n_samples <= 0) {
        return nullptr;
    }

    return reinterpret_cast<argus_bitmap_t*>(mtmd_bitmap_init_from_audio((size_t)n_samples, pcm_data));
}

argus_bitmap_t * argus_bitmap_load_file(argus_multimodal_t * mctx, const char * path, bool placeholder) {
    if (!mctx || !path) {
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
}

argus_bitmap_t * argus_bitmap_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, bool placeholder) {
    if (!mctx || !buffer || size <= 0) {
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
}

void argus_bitmap_free(argus_bitmap_t * bitmap) {
    if (bitmap) {
        mtmd_bitmap_free(reinterpret_cast<mtmd_bitmap*>(bitmap));
    }
}

argus_video_t * argus_video_load_file(argus_multimodal_t * mctx, const char * path, float fps_target, int64_t timestamp_interval_ms) {
    if (!mctx || !path) {
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
}

argus_video_t * argus_video_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, float fps_target, int64_t timestamp_interval_ms) {
    if (!mctx || !buffer || size <= 0) {
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
}

void argus_video_free(argus_video_t * video) {
    if (video) {
        if (video->video) {
            mtmd_helper_video_free(video->video);
        }
        delete video;
    }
}

int32_t argus_video_read_next(argus_video_t * video, argus_bitmap_t ** out_bitmap, char * out_text, int32_t max_chars) {
    if (!video || !out_bitmap || !out_text || max_chars <= 0) {
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
            // Verification: Checked tools/mtmd/mtmd-helper.cpp; mtmd_helper_video_read_next 
            // uses standard strdup() for out_text. Calling standard free() here is correct.
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
}

argus_input_chunks_t * argus_input_chunks_init(void) {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    argus_input_chunks_t * argus_chunks = new argus_input_chunks();
    argus_chunks->chunks = chunks;
    return argus_chunks;
}

void argus_input_chunks_free(argus_input_chunks_t * chunks) {
    if (chunks) {
        if (chunks->chunks) {
            mtmd_input_chunks_free(chunks->chunks);
        }
        delete chunks;
    }
}

int32_t argus_multimodal_tokenize(
    argus_multimodal_t * mctx,
    argus_input_chunks_t * output,
    const char * text,
    bool add_bos,
    const argus_bitmap_t ** bitmaps,
    int32_t n_bitmaps) {
    if (!mctx || !output || !text || (n_bitmaps > 0 && !bitmaps)) {
        return -1;
    }

    mtmd_input_text input_text;
    input_text.text = text;
    input_text.add_special = add_bos;
    input_text.parse_special = true;

    std::vector<const mtmd_bitmap *> mtmd_bitmaps(n_bitmaps);
    for (int32_t i = 0; i < n_bitmaps; ++i) {
        if (!bitmaps[i]) {
            return -1;
        }
        mtmd_bitmaps[i] = reinterpret_cast<const mtmd_bitmap*>(bitmaps[i]);
    }

    return mtmd_tokenize(mctx->ctx, output->chunks, &input_text, mtmd_bitmaps.data(), (size_t)n_bitmaps);
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
    if (!mctx || !ctx || !chunks || !out_new_n_past) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

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
    return res;
}

} // extern "C"
