#include "native_test_hooks.h"
#include "argus_dsp.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <climits>
#include <cstring>

#define CHECK(x) do { if (!(x)) { std::cerr << "Failed: " #x << " line " << __LINE__ << '\n'; std::abort(); } } while (0)
namespace {
const void * held_resource = nullptr;
int held_event = 0;
std::atomic<bool> entered{false};
std::atomic<bool> released{false};
std::atomic<bool> contended{false};
std::atomic<int> frees{0};
std::atomic<int> model_frees{0};
void observe(int event, const void * resource) {
    if (event == 6) { ++frees; return; }
    if (event == 8) { ++model_frees; return; }
    if (event >= 10) contended.store(true, std::memory_order_release);
    if (event == held_event && resource == held_resource &&
        !entered.exchange(true, std::memory_order_acq_rel)) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!released.load(std::memory_order_acquire)) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
void arm(int event, const void * resource) {
    held_event = event; held_resource = resource;
    entered.store(false, std::memory_order_relaxed);
    released.store(false, std::memory_order_relaxed);
    contended.store(false, std::memory_order_relaxed);
}
void await(const std::atomic<bool> & condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!condition.load(std::memory_order_acquire)) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
void release() {
    released.store(true, std::memory_order_release);
}
}
int main() {
    const float frames[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float output[] = {-99, -1, -1, -1, -1, -1, -1, -99};
    CHECK(argus_overlap_add(frames, 8, 6, 4, 2, 1, output + 1));
    const float expected[] = {-99, 2, 8, 10, 7, 8, 0, -99};
    for (int i = 0; i < 8; ++i) CHECK(output[i] == expected[i]);
    CHECK(!argus_overlap_add(frames, 7, 6, 4, 2, 1, output + 1));
    CHECK(!argus_overlap_add(frames, 8, -1, 4, 2, 1, output + 1));
    for (int i = 0; i < 8; ++i) CHECK(output[i] == expected[i]);
    const uint8_t pixel[] = {0, 0, 0};
    CHECK(argus_bitmap_from_rgb(0, 1, pixel) == nullptr);
    CHECK(argus_bitmap_from_rgb(UINT32_MAX, UINT32_MAX, pixel) == nullptr);
    char bounded[7]; std::memset(bounded, 0x55, sizeof(bounded));
    const int required = argus_build_info_copy(nullptr, 0);
    CHECK(required > 64);
    CHECK(argus_build_info_copy(bounded + 1, 3) == required);
    CHECK(bounded[0] == 0x55 && bounded[1] == 'v' && bounded[2] == 'e' && bounded[3] == 0 && bounded[4] == 0x55);
    CHECK(argus_build_info_copy(nullptr, 1) == -1 && argus_build_info_copy(bounded, -1) == -1);
    CHECK(argus_backend_init(nullptr));
    argus_model_params_t mp{}; mp.model_path = "tests/data/tiny.gguf";
    auto model = argus_model_load(&mp); CHECK(model);
    argus_context_params_t cp{}; cp.context_length = 128; cp.cpu_threads = 2; cp.n_seq_max = 1;
    auto first = argus_context_init(model, &cp);
    auto second = argus_context_init(model, &cp); CHECK(first && second);
    auto projector = argus_test_projector(model);
    auto independent = argus_test_projector(model);
    auto chunks = argus_test_chunks(); CHECK(projector && independent && chunks);
    // Both public C decode entry points preserve identical execution state.
    const int32_t initial_tokens[] = {1, 4, 5};
    argus_token_batch_t legacy{}; legacy.tokens = initial_tokens; legacy.n_tokens = 3; legacy.request_logits = true;
    CHECK(argus_decode_batch(first, &legacy) == 0);
    CHECK(argus_decode_tokens(second, initial_tokens, 3, 0, 0, true, nullptr) == 0);
    CHECK(argus_kv_cache_seq_pos_max(first, 0) == argus_kv_cache_seq_pos_max(second, 0));
    const int sampled = argus_sample_token(first, 0, 0, 1);
    CHECK(sampled >= 0 && sampled == argus_sample_token(second, 0, 0, 1));
    argus_kv_cache_clear_slot(first, 0, -1, -1); argus_kv_cache_clear_slot(second, 0, -1, -1);
    auto foreign_model = argus_model_load(&mp); CHECK(foreign_model);
    auto foreign_context = argus_context_init(foreign_model, &cp); CHECK(foreign_context);
    int unchanged_position = 93;
    CHECK(argus_eval_multimodal_chunks(projector, foreign_context, chunks, 0, 0, 8, false, &unchanged_position) < 0);
    CHECK(unchanged_position == 93 && argus_kv_cache_seq_pos_max(foreign_context, 0) == -1);
    CHECK(argus_multimodal_tokenize(projector, chunks, "", false, nullptr, -1) < 0);
    argus_context_free(foreign_context); argus_model_free(foreign_model);
    argus_test_set_observer(observe);
    argus_multimodal_params_t mm{}; mm.mmproj_path = "tests/data/tiny-mmproj.gguf";
    argus_test_fail_after_retain(true);
    CHECK(argus_multimodal_init(model, &mm) == nullptr);
    argus_test_fail_after_retain(false);
    CHECK(model_frees == 0);
    arm(3, projector);
    int pos1 = -1, pos2 = -1;
    std::thread a([&] { CHECK(argus_eval_multimodal_chunks(projector, first, chunks, 0, 0, 8, false, &pos1) == 0); });
    await(entered);
    std::atomic<bool> done{false};
    std::thread b([&] { CHECK(argus_eval_multimodal_chunks(projector, second, chunks, 0, 0, 8, false, &pos2) == 0); done = true; });
    await(contended); CHECK(!done);
    // Distinct projector + context progresses while the shared projector is held.
    int independent_pos = -1;
    CHECK(argus_eval_multimodal_chunks(independent, second, chunks, 0, 0, 8, false, &independent_pos) == 0);
    CHECK(independent_pos == 1);
    release(); a.join(); b.join(); CHECK(pos1 == 1 && pos2 == 1 && done);

    arm(3, projector);
    std::thread evaluation([&] { CHECK(argus_eval_multimodal_chunks(projector, first, chunks, 0, 0, 8, false, &pos1) == 0); });
    await(entered);
    const int32_t tokens[] = {1, 4, 5}; done = false;
    std::thread decode([&] { CHECK(argus_decode_tokens(first, tokens, 3, 0, 0, true, nullptr) == 0); done = true; });
    await(contended); CHECK(!done); release(); evaluation.join(); decode.join();
    CHECK(argus_kv_cache_seq_pos_max(first, 0) == 2);

    // Fail after pre-evaluation invalidation: caller output is not published.
    argus_test_set_eval_result(-7); int sentinel = 93;
    CHECK(argus_eval_multimodal_chunks(projector, first, chunks, 0, 0, 8, true, &sentinel) == -7);
    CHECK(sentinel == 93);
    CHECK(argus_eval_multimodal_chunks(projector, first, chunks, INT_MAX, 0, 8, false, &sentinel) < 0);
    CHECK(argus_eval_multimodal_chunks(projector, first, chunks, 0, 0, INT_MAX, false, &sentinel) < 0);
    CHECK(sentinel == 93 && argus_kv_cache_seq_pos_max(first, 0) == 2);
    CHECK(argus_sample_token(first, 0, 0, 1) < 0);
    CHECK(argus_eval_multimodal_chunks(projector, first, chunks, -1, 0, 8, false, &sentinel) < 0);
    CHECK(sentinel == 93);
    argus_test_set_eval_result(0);

    auto video = argus_test_video(projector, 100);
    argus_multimodal_free(projector); CHECK(frees == 0);
    std::mutex values_mutex; std::set<int> values;
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) readers.emplace_back([&] {
        char text[32]; argus_bitmap_t * bitmap = nullptr;
        for (;;) {
            int result = argus_video_read_next(video, &bitmap, text, sizeof(text));
            if (result == -1) break;
            CHECK(result == 0 && bitmap == nullptr);
            int value = std::stoi(text);
            std::lock_guard<std::mutex> lock(values_mutex);
            CHECK(values.insert(value).second);
        }
    });
    for (auto & reader : readers) reader.join();
    CHECK(values.size() == 100 && *values.begin() == 0 && *values.rbegin() == 99);
    argus_video_free(video); CHECK(frees == 1);
    argus_multimodal_free(independent); CHECK(frees == 2);
    argus_input_chunks_free(chunks);
    argus_context_free(first); argus_context_free(second); argus_model_free(model);
    CHECK(model_frees == 1);
    argus_test_set_observer(nullptr);
    argus_backend_free();
}
