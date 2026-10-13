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

    // Multithreaded logging continuation isolation test:
    // Threads 0 & 2 emit WARN lines followed by CONT chunks (must be received).
    // Threads 1 & 3 emit DEBUG lines followed by CONT chunks (must be suppressed).
    // Zero DEBUG lines or DEBUG continuations should leak into the callback,
    // and zero WARN lines or WARN continuations should be dropped.
    {
        struct LogCounters {
            std::atomic<int> warn_lines{0};
            std::atomic<int> warn_conts{0};
            std::atomic<int> debug_lines{0};
            std::atomic<int> debug_conts{0};
        } counters;

        argus_set_log_level(ARGUS_LOG_WARN);
        argus_set_log_callback([](argus_log_level_t level, const char * text, void * user_data) {
            auto * c = static_cast<LogCounters *>(user_data);
            if (!text) return;
            if (level == ARGUS_LOG_WARN) {
                c->warn_lines.fetch_add(1, std::memory_order_relaxed);
            } else if (level == ARGUS_LOG_CONT) {
                c->warn_conts.fetch_add(1, std::memory_order_relaxed);
            } else if (level == ARGUS_LOG_DEBUG) {
                c->debug_lines.fetch_add(1, std::memory_order_relaxed);
            }
        }, &counters);

        const int iterations = 1000;
        std::vector<std::thread> log_workers;
        for (int t = 0; t < 4; ++t) {
            log_workers.emplace_back([t, iterations]() {
                for (int i = 0; i < iterations; ++i) {
                    if (t % 2 == 0) {
                        // Emit WARN (level 3 in ggml) and CONT (level 5 in ggml)
                        argus_test_emit_log(3, "Warning line from worker\n");
                        argus_test_emit_log(5, "Continuation chunk of warning\n");
                    } else {
                        // Emit DEBUG (level 1 in ggml) and CONT (level 5 in ggml)
                        argus_test_emit_log(1, "Debug line from worker\n");
                        argus_test_emit_log(5, "Continuation chunk of debug\n");
                    }
                }
            });
        }
        for (auto & w : log_workers) w.join();

        argus_set_log_callback(nullptr, nullptr);

        // Exactly 2 workers * iterations WARN lines and WARN continuation chunks
        CHECK(counters.warn_lines.load() == 2 * iterations);
        CHECK(counters.warn_conts.load() == 2 * iterations);
        CHECK(counters.debug_lines.load() == 0);
        CHECK(counters.debug_conts.load() == 0);
    }

    // Test 6: Asynchronous callback replacement lifetime contract & user_data integrity
    {
        std::atomic<bool> a_entered{false};
        std::atomic<bool> a_release{false};
        std::atomic<bool> a_completed{false};
        std::atomic<int> b_count{0};

        struct CallbackAData {
            std::atomic<bool> * entered;
            std::atomic<bool> * release;
            std::atomic<bool> * completed;
            int sentinel = 0x12345678;
        } data_a{&a_entered, &a_release, &a_completed};

        // 1. Register callback A with user_data containing a verification sentinel
        argus_set_log_callback([](argus_log_level_t level, const char * text, void * user_data) {
            auto * d = static_cast<CallbackAData *>(user_data);
            CHECK(d != nullptr);
            CHECK(d->sentinel == 0x12345678);
            d->entered->store(true, std::memory_order_release);
            while (!d->release->load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            CHECK(d->sentinel == 0x12345678);
            d->completed->store(true, std::memory_order_release);
        }, &data_a);

        // 2. Worker thread emits log, capturing callback A and entering it
        std::thread worker([&]() {
            argus_test_emit_log(3, "Message dispatched to callback A\n");
        });

        // 3. Wait until callback A is actively executing in worker thread
        while (!a_entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        // 4. Concurrently replace callback A with callback B while A is still executing
        argus_set_log_callback([](argus_log_level_t level, const char * text, void * user_data) {
            auto * count = static_cast<std::atomic<int> *>(user_data);
            count->fetch_add(1, std::memory_order_relaxed);
        }, &b_count);

        // 5. Assert replacement returned while callback A is STILL executing in worker
        CHECK(!a_completed.load(std::memory_order_acquire));
        CHECK(argus_log_in_flight_count() > 0);

        // 6. Release callback A and wait for it to complete
        a_release.store(true, std::memory_order_release);
        worker.join();

        CHECK(a_completed.load(std::memory_order_acquire));

        // 7. Subsequent log dispatches must route to callback B
        argus_test_emit_log(3, "Message dispatched to callback B\n");
        CHECK(b_count.load() == 1);

        argus_set_log_callback(nullptr, nullptr);
    }

    // Multi-threaded concurrent argus_eval_embeddings stress test on a shared context
    {
        argus_model_params_t emp{}; emp.model_path = "tests/data/tiny-embed.gguf";
        auto emodel = argus_model_load(&emp);
        if (emodel) {
            argus_context_params_t ecp{};
            ecp.context_length = 256;
            ecp.cpu_threads = 2;
            ecp.n_seq_max = 2;
            ecp.embeddings = true;
            auto ectx = argus_context_init(emodel, &ecp);
            CHECK(ectx != nullptr);

            static const float EXPECTED_A[32] = {
                0.012607f, 0.008028f, 0.004507f, -0.023824f, 0.000028f, -0.013474f, 0.001769f, -0.006316f,
                -0.000113f, 0.000781f, 0.000226f, -0.019436f, -0.001485f, 0.004716f, -0.013539f, 0.005747f,
                -0.004355f, -0.003777f, -0.001283f, -0.001102f, 0.000275f, -0.000012f, 0.005550f, -0.016983f,
                0.025341f, 0.001413f, -0.002965f, 0.000518f, 0.002365f, -0.010931f, 0.030305f, -0.000967f
            };
            static const float EXPECTED_B[32] = {
                -0.003486f, 0.024704f, -0.008348f, -0.010331f, -0.000755f, -0.016964f, 0.002638f, -0.001489f,
                0.000092f, -0.000260f, -0.002435f, -0.005751f, -0.002582f, -0.000795f, -0.012421f, 0.004681f,
                -0.007213f, -0.020193f, -0.000497f, 0.000980f, 0.000381f, -0.000168f, 0.020802f, -0.018449f,
                0.031088f, -0.000668f, -0.000664f, 0.000861f, 0.013751f, -0.011461f, 0.019753f, 0.007501f
            };

            const int iterations = 10;
            std::atomic<bool> race_failed{false};

            auto worker_a = [&]() {
                const int32_t tokens_a[] = {1, 4, 5};
                float out[32];
                for (int i = 0; i < iterations; ++i) {
                    int32_t written = argus_eval_embeddings(ectx, tokens_a, 3, 0, out, 32);
                    if (written != 32) { race_failed.store(true); break; }
                    for (int j = 0; j < 32; ++j) {
                        if (std::abs(out[j] - EXPECTED_A[j]) > 1e-4f) { race_failed.store(true); break; }
                    }
                }
            };

            auto worker_b = [&]() {
                const int32_t tokens_b[] = {1, 4, 6};
                float out[32];
                for (int i = 0; i < iterations; ++i) {
                    int32_t written = argus_eval_embeddings(ectx, tokens_b, 3, 1, out, 32);
                    if (written != 32) { race_failed.store(true); break; }
                    for (int j = 0; j < 32; ++j) {
                        if (std::abs(out[j] - EXPECTED_B[j]) > 1e-4f) { race_failed.store(true); break; }
                    }
                }
            };

            std::thread t1(worker_a);
            std::thread t2(worker_b);
            t1.join();
            t2.join();

            CHECK(!race_failed.load());

            argus_context_free(ectx);
            argus_model_free(emodel);
        }
    }

    argus_backend_free();
}
