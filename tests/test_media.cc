#include "libargus.h"
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::cerr << "Failed: " #x << " line " << __LINE__ << '\n'; std::abort(); } } while (0)
int main(int argc, char **) {
    CHECK(argus_backend_init(nullptr));
    argus_model_params_t mp{}; mp.model_path = "tests/data/tiny.gguf";
    auto model = argus_model_load(&mp); CHECK(model);
    argus_context_params_t cp{}; cp.context_length = 128; cp.cpu_threads = 2; cp.n_seq_max = 1;
    auto ctx = argus_context_init(model, &cp); CHECK(ctx);
    argus_multimodal_params_t params{};
    params.mmproj_path = "tests/data/tiny-mmproj.gguf"; params.cpu_threads = 2;
    auto projector = argus_multimodal_init(model, &params); CHECK(projector);
    CHECK(argus_multimodal_support_vision(projector));
    std::vector<uint8_t> rgb(16 * 16 * 3, 127);
    auto bitmap = argus_bitmap_from_rgb(16, 16, rgb.data()); CHECK(bitmap);
    auto chunks = argus_input_chunks_init(); CHECK(chunks);
    const argus_bitmap_t * images[] = {bitmap};
    CHECK(argus_multimodal_tokenize(projector, chunks, "<__media__>", false, images, 1) == 0);
    int pos = -1;
    CHECK(argus_eval_multimodal_chunks(projector, ctx, chunks, 0, 0, 8, true, &pos) == 0);
    CHECK(pos == 4 && argus_kv_cache_seq_pos_max(ctx, 0) == 3);
    CHECK(argus_sample_token(ctx, 0, 0, 1) < 0);
    CHECK(argus_multimodal_tokenize(projector, chunks, "<__media__></s>", false, images, 1) == 0);
    int next = -1;
    CHECK(argus_eval_multimodal_chunks(projector, ctx, chunks, pos, 0, 8, true, &next) == 0);
    CHECK(next > pos + 4 && argus_kv_cache_seq_pos_max(ctx, 0) == next - 1);
    CHECK(argus_sample_token(ctx, 0, 0, 1) >= 0);
    argus_input_chunks_free(chunks); argus_bitmap_free(bitmap);
    if (argc > 1) {
        auto video = argus_video_load_file(projector, "tests/data/tiny-video.y4m", 2.0f, 0); CHECK(video);
        // Native iterator retains the projector after its public owner releases it.
        argus_multimodal_free(projector); projector = nullptr;
        int frames = 0; char text[128];
        for (;;) {
            bitmap = nullptr;
            int result = argus_video_read_next(video, &bitmap, text, sizeof(text));
            if (result == -1) break;
            CHECK(result == 0 || result == 1);
            if (bitmap) { ++frames; argus_bitmap_free(bitmap); }
        }
        CHECK(frames == 3);
        argus_video_free(video);
    }
    if (projector) argus_multimodal_free(projector);
    argus_context_free(ctx); argus_model_free(model); argus_backend_free();
}
