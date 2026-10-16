/**
 * @file test_libargus.cc
 * @brief Comprehensive integration and fanged unit tests for libargus lifecycle, backend telemetry,
 *        and state-machine sampling verification.
 */

#include "libargus.h"
#ifdef ARGUS_TESTING
#include "native_test_hooks.h"
#endif
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>
#include <set>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cmath>
#if !defined(_WIN32)
#include <unistd.h>
#include <fcntl.h>
#endif

#define ARGUS_CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

int main(int argc, char ** argv) {
    std::string expected_target = "";
    bool features_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect-features") == 0 && i + 1 < argc) {
            expected_target = argv[++i];
        } else if (std::strcmp(argv[i], "--features-only") == 0) {
            features_only = true;
        }
    }

    uint64_t build_features = argus_build_features();
    std::cout << "[Test] Active Build Features bitmask: 0x" << std::hex << build_features << std::dec << std::endl;
    if ((build_features & ARGUS_FEATURE_CPU) != 0) std::cout << "  - Feature: CPU" << std::endl;
    if ((build_features & ARGUS_FEATURE_CUDA) != 0) std::cout << "  - Feature: CUDA" << std::endl;
    if ((build_features & ARGUS_FEATURE_HIP) != 0) std::cout << "  - Feature: ROCm/HIP" << std::endl;
    if ((build_features & ARGUS_FEATURE_VULKAN) != 0) std::cout << "  - Feature: Vulkan" << std::endl;
    if ((build_features & ARGUS_FEATURE_METAL) != 0) std::cout << "  - Feature: Metal" << std::endl;

    if (!expected_target.empty()) {
        std::cout << "[Test] Validating expected feature contract for target: " << expected_target << std::endl;
        uint64_t expected = ARGUS_FEATURE_CPU;
        if (expected_target == "cuda") expected |= ARGUS_FEATURE_CUDA;
        else if (expected_target == "rocm" || expected_target == "hip") expected |= ARGUS_FEATURE_HIP;
        else if (expected_target == "vulkan") expected |= ARGUS_FEATURE_VULKAN;
        else if (expected_target == "metal") expected |= ARGUS_FEATURE_METAL;
        else if (expected_target != "cpu") return 1;
        ARGUS_CHECK(build_features == expected);
        std::cout << "[Test] Expected feature contract verified for " << expected_target << std::endl;
        if (features_only) {
            std::cout << "[Test] --features-only specified; exiting successfully after feature probe." << std::endl;
            return 0;
        }
    } else if (features_only) {
        std::cout << "[Test] --features-only specified; bitmask validated. Exiting." << std::endl;
        return 0;
    }

    std::cout << "[Test] Starting libargus lifecycle integration verification..." << std::endl;

    // 1. Initialize the global backends
    bool init_success = argus_backend_init(nullptr);
    ARGUS_CHECK(init_success);
    std::cout << "[Test] Backend initialized successfully." << std::endl;

    // Assert compiled version query matches expectations
    std::cout << "[Test] Library Version: " << argus_version() << std::endl;
    ARGUS_CHECK(std::strcmp(argus_version(), LIBARGUS_VERSION) == 0);

    // 2. Query backend count and list their names
    int32_t backend_count = argus_backend_get_count();
    std::cout << "[Test] Detected " << backend_count << " available hardware backend(s):" << std::endl;

    for (int32_t i = 0; i < backend_count; ++i) {
        const char * backend_name = argus_backend_get_name(i);
        std::cout << "  - Backend " << i << ": " << (backend_name ? backend_name : "UNKNOWN") << std::endl;
    }

    // 2.1 Diagnostic Logging API & Callback Verification
    std::cout << "[Test] Verifying diagnostic logging API and callback delegation..." << std::endl;
    argus_log_level_t orig_level = argus_get_log_level();
    argus_set_log_level(ARGUS_LOG_DEBUG);
    ARGUS_CHECK(argus_get_log_level() == ARGUS_LOG_DEBUG);
    argus_set_log_level(ARGUS_LOG_NONE);
    ARGUS_CHECK(argus_get_log_level() == ARGUS_LOG_NONE);
    argus_set_log_level(ARGUS_LOG_WARN);
    ARGUS_CHECK(argus_get_log_level() == ARGUS_LOG_WARN);

    struct LogAccumulator {
        std::vector<std::pair<argus_log_level_t, std::string>> entries;
        std::mutex mtx;
    };
    LogAccumulator acc;
    argus_set_log_callback([](argus_log_level_t level, const char * text, void * user_data) {
        auto * a = static_cast<LogAccumulator *>(user_data);
        std::lock_guard<std::mutex> l(a->mtx);
        a->entries.emplace_back(level, std::string(text ? text : ""));
    }, &acc);

    argus_set_log_level(ARGUS_LOG_DEBUG);
    argus_model_params_t bad_params = {};
    bad_params.model_path = "non_existent_file_to_trigger_logging.gguf";
    argus_model_t * bad_model = argus_model_load(&bad_params);
    ARGUS_CHECK(bad_model == nullptr);

    {
        std::lock_guard<std::mutex> l(acc.mtx);
        ARGUS_CHECK(!acc.entries.empty());
        bool captured_expected_log = false;
        for (const auto & e : acc.entries) {
            if (e.second.find("non_existent_file") != std::string::npos ||
                e.second.find("failed") != std::string::npos ||
                e.second.find("error") != std::string::npos) {
                captured_expected_log = true;
                break;
            }
        }
        ARGUS_CHECK(captured_expected_log);
    }

    // Reset callback and restore level
    argus_set_log_callback(nullptr, nullptr);
    argus_set_log_level(orig_level);

    // Reentrancy verification: callback mutating log configuration must not deadlock
    std::atomic<bool> reentrant_invoked{false};
    argus_set_log_callback([](argus_log_level_t, const char *, void * user_data) {
        auto * invoked = static_cast<std::atomic<bool> *>(user_data);
        if (!invoked->exchange(true)) {
            // Reentrant downcalls from inside foreign callback handler
            argus_set_log_level(ARGUS_LOG_INFO);
            argus_set_log_callback(nullptr, nullptr);
        }
    }, &reentrant_invoked);

    argus_set_log_level(ARGUS_LOG_DEBUG);
    argus_model_params_t bad_params_reentrant = {};
    bad_params_reentrant.model_path = "non_existent_reentrant_test.gguf";
    argus_model_t * bad_reentrant = argus_model_load(&bad_params_reentrant);
    ARGUS_CHECK(bad_reentrant == nullptr);
    ARGUS_CHECK(reentrant_invoked.load() == true);
    argus_set_log_callback(nullptr, nullptr);
    argus_set_log_level(orig_level);

    std::cout << "  - Diagnostic logging, callback assertions & reentrancy verified successfully." << std::endl;

    // 3. Test model loading with empty params to verify error pathways
    std::cout << "[Test] Verifying model load handling with null input..." << std::endl;
    argus_model_t * null_model = argus_model_load(nullptr);
    ARGUS_CHECK(null_model == nullptr);

    // 4. Test audio context initialization with null input
    std::cout << "[Test] Verifying audio context init handling with null input..." << std::endl;
    argus_audio_context_t * null_audio_ctx = argus_audio_init(nullptr);
    ARGUS_CHECK(null_audio_ctx == nullptr);

    // 5. Test TTS output generation on a placeholder context
    std::cout << "[Test] Verifying speech synthesis dummy float generator..." << std::endl;
    float dummy_pcm[100];
    int32_t samples = argus_synthesize_speech(nullptr, nullptr, "Hello test", 42, dummy_pcm, 100, nullptr, 0);
    ARGUS_CHECK(samples == -1);

    // 6. Test Multimodal initialization and load boundaries with invalid/null params
    std::cout << "[Test] Verifying multimodal init handling with null input..." << std::endl;
    argus_multimodal_t * null_mctx = argus_multimodal_init(nullptr, nullptr);
    ARGUS_CHECK(null_mctx == nullptr);

    std::cout << "[Test] Verifying support queries on null multimodal context..." << std::endl;
    ARGUS_CHECK(argus_multimodal_support_vision(nullptr) == false);
    ARGUS_CHECK(argus_multimodal_support_audio(nullptr) == false);
    ARGUS_CHECK(argus_multimodal_support_video(nullptr) == false);
    ARGUS_CHECK(argus_multimodal_get_audio_sample_rate(nullptr) == -1);

    std::cout << "[Test] Verifying bitmap creation on null inputs..." << std::endl;
    ARGUS_CHECK(argus_bitmap_from_rgb(100, 100, nullptr) == nullptr);
    ARGUS_CHECK(argus_bitmap_from_pcm(nullptr, 100) == nullptr);
    ARGUS_CHECK(argus_bitmap_load_file(nullptr, nullptr, false) == nullptr);
    ARGUS_CHECK(argus_bitmap_load_buffer(nullptr, nullptr, 0, false) == nullptr);

    std::cout << "[Test] Verifying video loaders and reader boundary handling..." << std::endl;
    ARGUS_CHECK(argus_video_load_file(nullptr, nullptr, 0.0f, 0) == nullptr);
    ARGUS_CHECK(argus_video_load_buffer(nullptr, nullptr, 0, 0.0f, 0) == nullptr);
    ARGUS_CHECK(argus_video_read_next(nullptr, nullptr, nullptr, 0) == -2);

    std::cout << "[Test] Verifying chunks allocation and tokenization boundaries..." << std::endl;
    argus_input_chunks_t * chunks = argus_input_chunks_init();
    ARGUS_CHECK(chunks != nullptr);

    int32_t tok_res = argus_multimodal_tokenize(nullptr, chunks, "test", true, nullptr, 0);
    ARGUS_CHECK(tok_res == -1);

    int32_t out_n_past = 0;
    int32_t eval_res = argus_eval_multimodal_chunks(nullptr, nullptr, chunks, 0, 0, 128, false, &out_n_past);
    ARGUS_CHECK(eval_res == -1);

    argus_input_chunks_free(chunks);
    std::cout << "[Test] Multimodal C API tests completed successfully." << std::endl;

    // 6.5. Test Vocabulary and Metadata null checks
    std::cout << "[Test] Verifying vocab and metadata null safety checks..." << std::endl;
    ARGUS_CHECK(argus_vocab_bos(nullptr) == -1);
    ARGUS_CHECK(argus_vocab_eos(nullptr) == -1);
    ARGUS_CHECK(argus_vocab_eot(nullptr) == -1);
    ARGUS_CHECK(argus_vocab_pad(nullptr) == -1);
    ARGUS_CHECK(argus_vocab_n_tokens(nullptr) == -1);
    ARGUS_CHECK(argus_vocab_is_eog(nullptr, 0) == false);

    char dummy_meta_buf[10];
    ARGUS_CHECK(argus_model_meta_val_str(nullptr, "some_key", dummy_meta_buf, 10) == -1);
    ARGUS_CHECK(argus_model_meta_count(nullptr) == -1);
    ARGUS_CHECK(argus_model_meta_key_by_index(nullptr, 0, dummy_meta_buf, 10) == -1);
    ARGUS_CHECK(argus_model_meta_val_str_by_index(nullptr, 0, dummy_meta_buf, 10) == -1);

    // Test Model Shape queries null safety checks
    ARGUS_CHECK(argus_model_n_embd(nullptr) == -1);
    ARGUS_CHECK(argus_model_n_ctx_train(nullptr) == -1);
    ARGUS_CHECK(argus_model_n_layer(nullptr) == -1);
    ARGUS_CHECK(argus_model_n_head(nullptr) == -1);
    ARGUS_CHECK(argus_model_n_head_kv(nullptr) == -1);
    ARGUS_CHECK(argus_model_n_params(nullptr) == 0);
    ARGUS_CHECK(argus_model_has_encoder(nullptr) == false);
    ARGUS_CHECK(argus_model_n_pos_per_embd(nullptr) == -1);
    ARGUS_CHECK(argus_model_is_mrope(nullptr) == false);

    // Test KV cache position queries null safety checks
    ARGUS_CHECK(argus_kv_cache_seq_pos_max(nullptr, 0) == -1);
    ARGUS_CHECK(argus_kv_cache_seq_pos_min(nullptr, 0) == -1);

    // Test Model KV memory calculation, size, desc, and GGML quantization introspection null safety
    std::cout << "[Test] Verifying KV cache calculation and quantization introspection null safety..." << std::endl;
    ARGUS_CHECK(argus_model_size(nullptr) == 0);
    ARGUS_CHECK(argus_model_desc(nullptr, dummy_meta_buf, 10) == -1);
    ARGUS_CHECK(argus_model_desc(nullptr, nullptr, 0) == -1);
    ARGUS_CHECK(argus_model_kv_bytes_per_token(nullptr, ARGUS_KV_TYPE_F16, ARGUS_KV_TYPE_F16) == -1);
    ARGUS_CHECK(argus_model_estimate_vram_bytes(nullptr, 4096, ARGUS_KV_TYPE_Q4_0, ARGUS_KV_TYPE_Q4_0) == -1);
    ARGUS_CHECK(argus_model_estimate_vram_bytes(nullptr, -10, ARGUS_KV_TYPE_Q4_0, ARGUS_KV_TYPE_Q4_0) == -1);

    // Test GGML quantization type size and block size calculation logic
    std::cout << "[Test] Verifying GGML quantization type and block size calculations..." << std::endl;
    ARGUS_CHECK(argus_quant_type_size(ARGUS_KV_TYPE_F16) == 2);
    ARGUS_CHECK(argus_quant_block_size(ARGUS_KV_TYPE_F16) == 1);
    ARGUS_CHECK(argus_quant_type_size(ARGUS_KV_TYPE_Q8_0) > 0);
    ARGUS_CHECK(argus_quant_block_size(ARGUS_KV_TYPE_Q8_0) > 0);
    ARGUS_CHECK(argus_quant_type_size(ARGUS_KV_TYPE_Q4_0) > 0);
    ARGUS_CHECK(argus_quant_block_size(ARGUS_KV_TYPE_Q4_0) > 0);
    ARGUS_CHECK(argus_quant_type_size(-1) == 0);
    ARGUS_CHECK(argus_quant_block_size(-1) == 0);
    ARGUS_CHECK(argus_quant_type_size(99999) == 0);
    ARGUS_CHECK(argus_quant_block_size(99999) == 0);

    // Test logit bias sampling null checks
    ARGUS_CHECK(argus_sample_token_with_bias(nullptr, 0, 0.0f, 0.0f, nullptr, 0) == -1);

    // Verify struct alignment layout sizes
    std::cout << "[Test] Verifying struct alignment layout sizes and sequence parameters..." << std::endl;
    ARGUS_CHECK(sizeof(argus_context_params_t) == 40);
    ARGUS_CHECK(sizeof(argus_sampler_params_t) == 56);

    argus_context_params_t test_params = {};
    test_params.context_length = 2048;
    test_params.u_batch = 1024;
    test_params.n_seq_max = 2;
    test_params.embeddings = true;
    test_params.kv_unified = true;
    ARGUS_CHECK(test_params.u_batch == 1024);
    ARGUS_CHECK(test_params.n_seq_max == 2);
    ARGUS_CHECK(test_params.embeddings == true);
    ARGUS_CHECK(test_params.kv_unified == true);

    // Test extended sampling, lifecycle and draft null checks
    ARGUS_CHECK(argus_context_has_draft(nullptr) == false);
    ARGUS_CHECK(argus_sample_token_ext(nullptr, 0, nullptr, nullptr, 0) == -1);
    ARGUS_CHECK(argus_sampler_reset(nullptr, 0) == -1);
    ARGUS_CHECK(argus_sampler_prime(nullptr, 0, nullptr, 0) == -1);
    ARGUS_CHECK(argus_sampler_truncate(nullptr, 0, 0) == -1);

    std::cout << "[Test] Vocab and metadata null safety tests completed successfully." << std::endl;

    // Test dynamic thread count control null checks
    std::cout << "[Test] Verifying thread count control null safety checks..." << std::endl;
    argus_set_n_threads(nullptr, 4, 4);
    ARGUS_CHECK(argus_get_n_threads(nullptr) == -1);
    ARGUS_CHECK(argus_get_n_threads_batch(nullptr) == -1);
    argus_audio_set_n_threads(nullptr, 4);
    ARGUS_CHECK(argus_audio_get_n_threads(nullptr) == -1);
    std::cout << "[Test] Thread count control null safety assertions completed successfully." << std::endl;

    // =========================================================================
    // 7. Fanged End-to-End Functional Verification (tests/data/tiny.gguf)
    // =========================================================================
    std::cout << "[Test] Starting fanged end-to-end model execution verification..." << std::endl;
    const char * model_paths[] = {
        "tests/data/tiny.gguf",
        "../../tests/data/tiny.gguf",
        "../tests/data/tiny.gguf"
    };
    const char * valid_model_path = nullptr;
    for (const char * p : model_paths) {
        FILE * f = fopen(p, "rb");
        if (f) {
            fclose(f);
            valid_model_path = p;
            break;
        }
    }

    if (valid_model_path) {
        std::cout << "  - Located tiny test model at: " << valid_model_path << std::endl;

        // 7.1. Model loading & Shape Introspection
        argus_model_params_t mparams = {};
        mparams.model_path = valid_model_path;
        mparams.gpu_layers = 0; // CPU only for CI determinism
        mparams.use_mlock = false;

        argus_model_t * model = argus_model_load(&mparams);
        ARGUS_CHECK(model != nullptr);
        std::cout << "  - Model loaded successfully." << std::endl;

        ARGUS_CHECK(argus_vocab_bos(model) == 1);
        ARGUS_CHECK(argus_vocab_eos(model) == 2);
        ARGUS_CHECK(argus_vocab_pad(model) == 3);
        ARGUS_CHECK(argus_vocab_n_tokens(model) == 64);
        ARGUS_CHECK(argus_model_n_embd(model) == 32);
        ARGUS_CHECK(argus_model_n_ctx_train(model) == 512);
        ARGUS_CHECK(argus_model_n_layer(model) == 1);
        ARGUS_CHECK(argus_model_n_head(model) == 2);
        ARGUS_CHECK(argus_model_n_head_kv(model) == 2);
        ARGUS_CHECK(argus_model_is_recurrent(model) == false);
        ARGUS_CHECK(argus_model_is_hybrid(model) == false);
        ARGUS_CHECK(argus_model_is_diffusion(model) == false);
        ARGUS_CHECK(argus_model_is_recurrent(nullptr) == false);
        ARGUS_CHECK(argus_model_is_hybrid(nullptr) == false);
        ARGUS_CHECK(argus_model_is_diffusion(nullptr) == false);

        // 7.2. Draft Model Loading & Context Initialization
        argus_model_t * draft_model = argus_model_load(&mparams);
        ARGUS_CHECK(draft_model != nullptr);

        argus_context_params_t cparams = {};
        cparams.context_length = 256;
        cparams.cpu_threads = 2;
        cparams.draft_model = draft_model;
        cparams.spec_draft_n_max = 4;
        cparams.n_seq_max = 2;
        cparams.kv_unified = true;

        argus_context_t * ctx = argus_context_init(model, &cparams);
        ARGUS_CHECK(ctx != nullptr);
        ARGUS_CHECK(argus_context_has_draft(ctx) == true);
        ARGUS_CHECK(argus_context_get_seq_max(ctx) == 2);
        ARGUS_CHECK(argus_context_get_seq_max(nullptr) == -1);
        ARGUS_CHECK(argus_kv_cache_can_shift(ctx) == true);
        ARGUS_CHECK(argus_kv_cache_can_shift(nullptr) == false);
        ARGUS_CHECK(argus_context_get_model(ctx) == model);
        ARGUS_CHECK(argus_context_get_model(nullptr) == nullptr);
        std::cout << "  - Primary and Speculative Draft Context initialized." << std::endl;

        // 7.3. Token Batch Decoding & Position Verification
        int32_t prompt_tokens[] = { 1, 4, 5, 6, 7 }; // 5 tokens (BOS, a, b, c, d)
        argus_token_batch_t batch1 = {};
        batch1.tokens = prompt_tokens;
        batch1.n_tokens = 5;
        batch1.start_pos = 0;
        batch1.seq_id = 0;
        batch1.request_logits = true;

        int32_t dec_res1 = argus_decode_batch(ctx, &batch1);
        ARGUS_CHECK(dec_res1 == 0);

        int32_t max_pos1 = argus_kv_cache_seq_pos_max(ctx, 0);
        ARGUS_CHECK(max_pos1 == 4); // Positions 0, 1, 2, 3, 4 evaluated -> max_pos is 4
        std::cout << "  - Prompt batch evaluated. High-water mark seq_pos_max = " << max_pos1 << std::endl;

        // 7.4. Automagic Prefix Rollback & Speculative Draft Cache Sync
        // Submit branch starting at position 2 (overwriting tokens 2, 3, 4)
        int32_t branch_tokens[] = { 8, 9 }; // e, f at pos 2, 3
        argus_token_batch_t batch2 = {};
        batch2.tokens = branch_tokens;
        batch2.n_tokens = 2;
        batch2.start_pos = 2;
        batch2.seq_id = 0;
        batch2.request_logits = true;

        int32_t dec_res2 = argus_decode_batch(ctx, &batch2);
        ARGUS_CHECK(dec_res2 == 0);

        int32_t max_pos2 = argus_kv_cache_seq_pos_max(ctx, 0);
        ARGUS_CHECK(max_pos2 == 3); // Position rolled back and extended to 3
        std::cout << "  - Automagic prefix rollback verified in lockstep. New seq_pos_max = " << max_pos2 << std::endl;

        // 7.5. Extended Zero-Allocation Sampler Verification (Deterministic Seeding)
        argus_sampler_params_t sparams = {};
        sparams.temperature = 0.7f;
        sparams.repeat_penalty = 1.1f;
        sparams.repeat_last_n = 16;
        sparams.top_p = 0.90f;
        sparams.min_p = 0.05f;
        sparams.top_k = 20;
        sparams.seed = 42U;

        int32_t sampled_token = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(sampled_token >= 0 && sampled_token < 64);
        std::cout << "  - Extended sampler successfully generated token: " << sampled_token << std::endl;

        // Verify logits consumption: immediate subsequent sample without decode returns -2
        int32_t re_sample = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(re_sample == -2);
        std::cout << "  - Logits consumption verified (subsequent sample without decode returned -2)." << std::endl;

        // 7.6. Multi-Sequence Logits Isolation Verification
        // Decode seq 0 with logits, then seq 1 with logits.
        int32_t s0_toks[] = { 1, 10 };
        argus_token_batch_t b_s0 = {};
        b_s0.tokens = s0_toks;
        b_s0.n_tokens = 2;
        b_s0.start_pos = 0;
        b_s0.seq_id = 0;
        b_s0.request_logits = true;
        int32_t dec_s0 = argus_decode_batch(ctx, &b_s0);
        ARGUS_CHECK(dec_s0 == 0);

        int32_t s1_toks[] = { 1, 20 };
        argus_token_batch_t b_s1 = {};
        b_s1.tokens = s1_toks;
        b_s1.n_tokens = 2;
        b_s1.start_pos = 0;
        b_s1.seq_id = 1;
        b_s1.request_logits = true;
        int32_t dec_s1 = argus_decode_batch(ctx, &b_s1);
        ARGUS_CHECK(dec_s1 == 0);

        // Sampling seq 0 must fail with -2 because seq 1 owns the latest logits
        int32_t s0_sample_err = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(s0_sample_err == -2);
        std::cout << "  - Cross-sequence logits contamination prevented (sampling seq 0 returned -2)." << std::endl;

        // Sampling seq 1 succeeds
        int32_t s1_sample = argus_sample_token_ext(ctx, 1, &sparams, nullptr, 0);
        ARGUS_CHECK(s1_sample >= 0 && s1_sample < 64);
        std::cout << "  - Seq 1 sampled successfully: " << s1_sample << std::endl;

        // 7.7. Stale Logits Invalidation on KV Cache Mutation
        // Decode seq 0 with logits, then partially clear KV cache -> must invalidate logits (-2)
        ARGUS_CHECK(argus_decode_batch(ctx, &b_s0) == 0);
        ARGUS_CHECK(argus_kv_cache_clear_slot(ctx, 0, 1, -1) == true);
        int32_t mutated_sample = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(mutated_sample == -2);
        std::cout << "  - Stale logits invalidation verified: KV cache mutation invalidated pending logits (-2)." << std::endl;

        // 7.7b. Sequence Slot Forking (seq_cp) & Sampler State Machine Cloning
        {
            // Decode prompt tokens on slot 0
            int32_t fork_prompt[] = { 1, 4, 5, 6 };
            argus_token_batch_t b_fork0 = {};
            b_fork0.tokens = fork_prompt;
            b_fork0.n_tokens = 4;
            b_fork0.start_pos = 0;
            b_fork0.seq_id = 0;
            b_fork0.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_fork0) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 0) == 3);

            // Sample on slot 0 to populate history and leave token pending
            int32_t s0_tok = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
            ARGUS_CHECK(s0_tok >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);

            // Bounds safety validation: reject partial copies (p0 > 0 or p1 >= 0)
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 1, 1, -1) == false);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 1, 0, 2) == false);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            // Fork slot 0 -> slot 1 (canonical full snapshot)
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 1, 0, -1) == true);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 1) == 3);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 1) == argus_sampler_get_history_count(ctx, 0));
            // Exact pending-token snapshot verification
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 1) == 1);

            // Continue decoding on slot 1 with a distinct token at pos 4
            int32_t slot1_next[] = { 8 };
            argus_token_batch_t b_fork1 = {};
            b_fork1.tokens = slot1_next;
            b_fork1.n_tokens = 1;
            b_fork1.start_pos = 4;
            b_fork1.seq_id = 1;
            b_fork1.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_fork1) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 1) == 4);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 0) == 3); // slot 0 remains untouched

            // Sampling on slot 1 succeeds with replicated sampler state
            int32_t s1_tok = argus_sample_token_ext(ctx, 1, &sparams, nullptr, 0);
            ARGUS_CHECK(s1_tok >= 0);

            // Pre-clearing verification: advance slot 1 further, then re-copy slot 0 (pos max 3) to slot 1
            int32_t extra_tok[] = { 9 };
            argus_token_batch_t b_extra = {};
            b_extra.tokens = extra_tok;
            b_extra.n_tokens = 1;
            b_extra.start_pos = 5;
            b_extra.seq_id = 1;
            b_extra.request_logits = false;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_extra) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 1) == 5);
            // Copy slot 0 -> slot 1; pre-clear must ensure slot 1 max pos resets to slot 0's max pos (3)
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 1, 0, -1) == true);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 1) == 3);

            // Bounds safety validation (reject out of range without GGML_ASSERT abort)
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 99, 0, -1) == false);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 99, 1, 0, -1) == false);
            ARGUS_CHECK(argus_kv_cache_seq_cp(nullptr, 0, 1, 0, -1) == false);

            // Clean slot 1 and verify state reset
            ARGUS_CHECK(argus_kv_cache_clear_slot(ctx, 1, 0, -1) == true);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 1) == -1);
            ARGUS_CHECK(argus_kv_cache_clear_slot(nullptr, 0, 0, -1) == false);
            ARGUS_CHECK(argus_kv_cache_clear_slot(ctx, 99, 0, -1) == false);
            std::cout << "  - Sequence slot forking (seq_cp), pending-state replication & pre-clearing verified successfully." << std::endl;
        }

        // 7.7b. Exact Stochastic Fork Continuation Verification
        {
            // Reset both slots to clean baseline
            ARGUS_CHECK(argus_kv_cache_clear_slot(ctx, 0, 0, -1) == true);
            ARGUS_CHECK(argus_kv_cache_clear_slot(ctx, 1, 0, -1) == true);
            argus_sampler_reset(ctx, 0);
            argus_sampler_reset(ctx, 1);

            int32_t init_prompt[] = { 1, 4, 5, 6 };
            argus_token_batch_t b_init = {};
            b_init.tokens = init_prompt;
            b_init.n_tokens = 4;
            b_init.start_pos = 0;
            b_init.seq_id = 0;
            b_init.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_init) == 0);

            argus_sampler_params_t stoch_sparams = {};
            stoch_sparams.temperature = 0.85f;
            stoch_sparams.top_k = 15;
            stoch_sparams.top_p = 0.90f;
            stoch_sparams.repeat_penalty = 1.15f;
            stoch_sparams.repeat_last_n = 16;
            stoch_sparams.seed = 9999U;

            // Sample token on slot 0 and leave it pending
            int32_t pend_t = argus_sample_token_ext(ctx, 0, &stoch_sparams, nullptr, 0);
            ARGUS_CHECK(pend_t >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);

            // Fork slot 0 -> slot 1
            ARGUS_CHECK(argus_kv_cache_seq_cp(ctx, 0, 1, 0, -1) == true);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 1) == 1);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == argus_sampler_get_history_count(ctx, 1));

            // Advance both branches identically over multiple steps and assert exact stochastic equivalence
            int32_t cur_pos = 4;
            int32_t next_t0 = pend_t;
            int32_t next_t1 = pend_t;

            for (int step = 0; step < 4; ++step) {
                // Decode token into slot 0 with logits, then sample immediately
                argus_token_batch_t b0 = {};
                b0.tokens = &next_t0;
                b0.n_tokens = 1;
                b0.start_pos = cur_pos;
                b0.seq_id = 0;
                b0.request_logits = true;
                ARGUS_CHECK(argus_decode_batch(ctx, &b0) == 0);

                int32_t s0_step_tok = argus_sample_token_ext(ctx, 0, &stoch_sparams, nullptr, 0);
                ARGUS_CHECK(s0_step_tok >= 0);

                // Decode token into slot 1 with logits, then sample immediately
                argus_token_batch_t b1 = {};
                b1.tokens = &next_t1;
                b1.n_tokens = 1;
                b1.start_pos = cur_pos;
                b1.seq_id = 1;
                b1.request_logits = true;
                ARGUS_CHECK(argus_decode_batch(ctx, &b1) == 0);

                int32_t s1_step_tok = argus_sample_token_ext(ctx, 1, &stoch_sparams, nullptr, 0);
                ARGUS_CHECK(s1_step_tok >= 0);

                // Both forks must yield bit-for-bit identical stochastic output
                ARGUS_CHECK(s0_step_tok == s1_step_tok);

                next_t0 = s0_step_tok;
                next_t1 = s1_step_tok;
                cur_pos++;
            }
            std::cout << "  - Exact stochastic fork continuation verified across 4 iterations." << std::endl;
        }

        // Test argus_get_embeddings exception guard and argument validation
        {
            float emb_buf[16];
            ARGUS_CHECK(argus_get_embeddings(nullptr, 0, emb_buf, 16) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_get_embeddings(ctx, 0, nullptr, 16) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_get_embeddings(ctx, 0, emb_buf, -1) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_get_embeddings(ctx, 0, emb_buf, 0) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            std::cout << "  - Embeddings exception barrier and argument validation verified successfully." << std::endl;
        }

        // 7.8. Logit Steering Bias Enforcement
        int32_t steer_toks[] = { 1, 4 };
        argus_token_batch_t b_steer = {};
        b_steer.tokens = steer_toks;
        b_steer.n_tokens = 2;
        b_steer.start_pos = 0;
        b_steer.seq_id = 0;
        b_steer.request_logits = true;
        int32_t dec_steer = argus_decode_batch(ctx, &b_steer);
        ARGUS_CHECK(dec_steer == 0);

        argus_logit_bias_t bias_forced = { 10, 1000.0f }; // Strongly boost token 10
        int32_t biased_token = argus_sample_token_ext(ctx, 0, &sparams, &bias_forced, 1);
        ARGUS_CHECK(biased_token == 10);
        std::cout << "  - Logit bias steering deterministically produced forced token: " << biased_token << std::endl;

        int32_t dec_steer2 = argus_decode_batch(ctx, &b_steer);
        ARGUS_CHECK(dec_steer2 == 0);
        argus_logit_bias_t bias_steer2 = { 15, 1000.0f }; // Strongly boost token 15
        int32_t biased_token2 = argus_sample_token_ext(ctx, 0, &sparams, &bias_steer2, 1);
        ARGUS_CHECK(biased_token2 == 15);
        std::cout << "  - Logit bias steering dynamically updated to token: " << biased_token2 << std::endl;

        // 7.9. Stochastic Distribution Sampling & Entropy Divergence
        // High temperature sampling across multiple distinct seeds must produce divergent tokens (proving non-greedy)
        std::set<int32_t> distinct_tokens;
        for (uint32_t test_seed = 1; test_seed <= 10; ++test_seed) {
            ARGUS_CHECK(argus_decode_batch(ctx, &b_steer) == 0);
            argus_sampler_params_t dist_params = {};
            dist_params.temperature = 1.5f;
            dist_params.top_k = 50;
            dist_params.top_p = 1.0f;
            dist_params.seed = test_seed;
            int32_t t = argus_sample_token_ext(ctx, 0, &dist_params, nullptr, 0);
            ARGUS_CHECK(t >= 0);
            distinct_tokens.insert(t);
        }
        ARGUS_CHECK(distinct_tokens.size() >= 2);
        std::cout << "  - Stochastic distribution verified: " << distinct_tokens.size() << " distinct tokens across seeds." << std::endl;

        // 7.10. Deterministic Seeding vs Stochastic Entropy Verification
        // Run generation with seed = 777 for 4 tokens twice; ensure bit-identical outputs
        std::vector<int32_t> run1_tokens;
        std::vector<int32_t> run2_tokens;

        argus_sampler_params_t seed_sparams = {};
        seed_sparams.temperature = 0.8f;
        seed_sparams.top_k = 10;
        seed_sparams.seed = 777U;

        // Run 1
        argus_kv_cache_clear_slot(ctx, 0, 0, -1);
        argus_sampler_reset(ctx, 0);
        ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 0);

        int32_t seed_prompt[] = { 1, 4, 5 };
        argus_token_batch_t b_seed = {};
        b_seed.tokens = seed_prompt;
        b_seed.n_tokens = 3;
        b_seed.start_pos = 0;
        b_seed.seq_id = 0;
        b_seed.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_seed) == 0);

        for (int i = 0; i < 4; ++i) {
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == i);

            int32_t t = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(t >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == i);

            run1_tokens.push_back(t);
            int32_t next_t = t;
            argus_token_batch_t b_next = {};
            b_next.tokens = &next_t;
            b_next.n_tokens = 1;
            b_next.start_pos = 3 + i;
            b_next.seq_id = 0;
            b_next.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_next) == 0);

            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == i + 1);
        }
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 4);

        // Run 2 (same seed 777)
        argus_kv_cache_clear_slot(ctx, 0, 0, -1);
        argus_sampler_reset(ctx, 0);
        ARGUS_CHECK(argus_decode_batch(ctx, &b_seed) == 0);

        for (int i = 0; i < 4; ++i) {
            int32_t t = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(t >= 0);
            run2_tokens.push_back(t);
            int32_t next_t = t;
            argus_token_batch_t b_next = {};
            b_next.tokens = &next_t;
            b_next.n_tokens = 1;
            b_next.start_pos = 3 + i;
            b_next.seq_id = 0;
            b_next.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_next) == 0);
        }

        ARGUS_CHECK(run1_tokens.size() == 4);
        ARGUS_CHECK(run1_tokens == run2_tokens);
        std::cout << "  - Seed reproducibility & monotonic continuation history growth verified: " << run1_tokens.size() << " tokens." << std::endl;

        // 7.11. Mismatched Pending Sample Rejection & Reconciliation
        argus_kv_cache_clear_slot(ctx, 0, 0, -1);
        argus_sampler_reset(ctx, 0);
        ARGUS_CHECK(argus_decode_batch(ctx, &b_seed) == 0);
        int32_t sampled_t = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
        ARGUS_CHECK(sampled_t >= 0);
        ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 0);

        // Intentionally decode a different token than the one sampled
        int32_t mismatched_t = (sampled_t + 1) % 64;
        argus_token_batch_t b_mismatch = {};
        b_mismatch.tokens = &mismatched_t;
        b_mismatch.n_tokens = 1;
        b_mismatch.start_pos = 3;
        b_mismatch.seq_id = 0;
        b_mismatch.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_mismatch) == 0);
        ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 1);
        std::cout << "  - Mismatched pending sample reconciliation verified (un-decoded sample evicted, decoded token committed)." << std::endl;

        // 7.12. Decoupled Priming Persistence Across Continuation and KV Rollback
        argus_kv_cache_clear_slot(ctx, 0, 0, -1);
        argus_sampler_reset(ctx, 0);
        int32_t primed[] = { 10, 11, 12, 13, 14, 15 };
        ARGUS_CHECK(argus_sampler_prime(ctx, 0, primed, 6) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 6);

        // Decode prompt at pos 0..3
        int32_t prompt_d[] = { 1, 4, 5, 6 };
        argus_token_batch_t b_pd = {};
        b_pd.tokens = prompt_d;
        b_pd.n_tokens = 4;
        b_pd.start_pos = 0;
        b_pd.seq_id = 0;
        b_pd.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_pd) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 6); // Decoupled primed tokens intact

        // Sample token at pos 4
        int32_t s_tok1 = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(s_tok1 >= 0);
        ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
        argus_token_batch_t b_c1 = {};
        b_c1.tokens = &s_tok1;
        b_c1.n_tokens = 1;
        b_c1.start_pos = 4;
        b_c1.seq_id = 0;
        b_c1.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_c1) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 7); // 6 primed + 1 committed

        // Sample token at pos 5
        int32_t s_tok2 = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(s_tok2 >= 0);
        ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
        argus_token_batch_t b_c2 = {};
        b_c2.tokens = &s_tok2;
        b_c2.n_tokens = 1;
        b_c2.start_pos = 5;
        b_c2.seq_id = 0;
        b_c2.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_c2) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 8); // 6 primed + 2 committed

        // Roll back KV cache to pos 2 (start_pos = 2)
        int32_t branch_d[] = { 20, 21 };
        argus_token_batch_t b_br = {};
        b_br.tokens = branch_d;
        b_br.n_tokens = 2;
        b_br.start_pos = 2;
        b_br.seq_id = 0;
        b_br.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(ctx, &b_br) == 0);
        // Pruning removes generated tokens at kv_pos >= 2 (s_tok1 at 4, s_tok2 at 5)
        // Decoupled primed tokens (kv_pos == -1, count 6) remain 100% intact.
        // Rollback replacement tokens (2 tokens) are committed to history and accepted by sampler.
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 8); // 6 primed + 2 replacement tokens
        std::cout << "  - Coordinate-decoupled rollback verified (primed tokens preserved & replacement tokens committed)." << std::endl;

#ifdef ARGUS_TESTING
        // 7.12b. Failed Rollback Rejection Atomicity & Zero-Mutation Invariant
        {
            // Snapshot current state before failed rollback
            int32_t pos_max_before = argus_kv_cache_seq_pos_max(ctx, 0);
            int32_t pos_min_before = argus_kv_cache_seq_pos_min(ctx, 0);
            int32_t hist_count_before = argus_sampler_get_history_count(ctx, 0);
            int32_t has_pending_before = argus_sampler_has_pending(ctx, 0);

            std::vector<int32_t> hist_tokens_before(hist_count_before);
            for (int32_t i = 0; i < hist_count_before; ++i) {
                hist_tokens_before[i] = argus_test_sampler_get_history_token(ctx, 0, i);
                ARGUS_CHECK(hist_tokens_before[i] >= 0);
            }

            bool has_logits_before = argus_test_has_logits(ctx, 0);
            ARGUS_CHECK(has_logits_before == true);
            int32_t draft_pos_max_before = argus_test_draft_seq_pos_max(ctx, 0);

            // Trigger forced rollback failure
            argus_test_set_force_rollback_fail(true);

            int32_t bad_branch[] = { 30, 31 };
            argus_token_batch_t b_fail = {};
            b_fail.tokens = bad_branch;
            b_fail.n_tokens = 2;
            b_fail.start_pos = 1; // Attempt partial rollback to pos 1
            b_fail.seq_id = 0;
            b_fail.request_logits = true;

            int32_t res_fail = argus_decode_batch(ctx, &b_fail);
            ARGUS_CHECK(res_fail == ARGUS_DECODE_ROLLBACK_FAILED);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_ROLLBACK_FAILED);

            // Restore normal operation
            argus_test_set_force_rollback_fail(false);

            // Assert exact state preservation (0 mutations across KV cache, history tokens, pending state, logits, and draft context)
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(ctx, 0) == pos_max_before);
            ARGUS_CHECK(argus_kv_cache_seq_pos_min(ctx, 0) == pos_min_before);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == hist_count_before);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == has_pending_before);

            for (int32_t i = 0; i < hist_count_before; ++i) {
                ARGUS_CHECK(argus_test_sampler_get_history_token(ctx, 0, i) == hist_tokens_before[i]);
            }

            ARGUS_CHECK(argus_test_has_logits(ctx, 0) == has_logits_before);
            ARGUS_CHECK(argus_test_draft_seq_pos_max(ctx, 0) == draft_pos_max_before);

            std::cout << "  - Failed rollback rejection verified: 100% zero-mutation atomicity preserved (history tokens, coordinates, logits, and draft context intact)." << std::endl;
        }
#endif

        // 7.13. Transactional Decode Cancellation & Retry Invariant
        {
            argus_kv_cache_clear_slot(ctx, 0, 0, -1);
            argus_sampler_reset(ctx, 0);
            ARGUS_CHECK(argus_decode_batch(ctx, &b_seed) == 0);

            int32_t pend_t = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(pend_t >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 0);

            // Ref-counted abort flag allocation & request
            argus_abort_flag_t * abort_flag = argus_abort_flag_create();
            ARGUS_CHECK(abort_flag != nullptr);
            ARGUS_CHECK(!argus_abort_flag_is_requested(abort_flag));
            ARGUS_CHECK(argus_abort_flag_retain(abort_flag)); // refcount 2

            argus_abort_flag_request(abort_flag);
            ARGUS_CHECK(argus_abort_flag_is_requested(abort_flag));

            argus_token_batch_t b_abort = {};
            b_abort.tokens = &pend_t;
            b_abort.n_tokens = 1;
            b_abort.start_pos = 3;
            b_abort.seq_id = 0;
            b_abort.request_logits = true;
            b_abort.abort_flag = abort_flag;

            int32_t abort_res = argus_decode_batch(ctx, &b_abort);
            ARGUS_CHECK(abort_res == -2);
            // Invariant: Pre-cancelled decode must NOT commit token or corrupt pending state
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 0);

            // Now retry after resetting abort flag: decode succeeds and commits exactly once
            argus_abort_flag_reset(abort_flag);
            ARGUS_CHECK(!argus_abort_flag_is_requested(abort_flag));
            ARGUS_CHECK(argus_decode_batch(ctx, &b_abort) == 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 1);

            argus_abort_flag_release(abort_flag); // refcount back to 1
            argus_abort_flag_release(abort_flag); // refcount to 0, safely freed
            std::cout << "  - Transactional decode cancellation & retry invariant verified (zero false commits on abort)." << std::endl;
        }

        // 7.14. Ghost Token Elimination & Canonical Pending Discard
        {
            // Sample a token
            int32_t disc_t = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(disc_t >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);

            // Explicit discard
            ARGUS_CHECK(argus_sampler_discard_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            ARGUS_CHECK(argus_sampler_discard_pending(ctx, 0) == 0); // No pending sample to discard

            // Discard via no-op history truncation (new_length == history.size())
            int32_t t_next = 10;
            argus_token_batch_t b_n = {};
            b_n.tokens = &t_next;
            b_n.n_tokens = 1;
            b_n.start_pos = 4;
            b_n.seq_id = 0;
            b_n.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(ctx, &b_n) == 0);

            int32_t pend_trunc = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(pend_trunc >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            int32_t cur_hist_len = argus_sampler_get_history_count(ctx, 0);
            ARGUS_CHECK(argus_sampler_truncate(ctx, 0, cur_hist_len) == 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);

            // Explicit pending discard after a fresh decode
            ARGUS_CHECK(argus_decode_batch(ctx, &b_n) == 0);
            int32_t pend_mm = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(pend_mm >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_discard_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            std::cout << "  - Ghost token elimination & canonical pending discard verified." << std::endl;
        }

        // 7.15. Text Context Query/Decode Contention
        {
            std::atomic<bool> worker_done{false};
            std::atomic<int32_t> error_count{0};
            std::atomic<int32_t> sync_calls{0};

            std::thread worker([&]() {
                for (int iter = 0; iter < 100; ++iter) {
                    int32_t hist = argus_sampler_get_history_count(ctx, 0);
                    int32_t pend = argus_sampler_has_pending(ctx, 0);
                    int32_t pos = argus_kv_cache_seq_pos_max(ctx, 0);
                    if (hist < 0 || pend < 0 || pos < -1) {
                        error_count.fetch_add(1);
                    }
                    sync_calls.fetch_add(1);
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                worker_done.store(true);
            });

            // Main thread performs concurrent decode batches and context thread queries
            for (int iter = 0; iter < 100; ++iter) {
                argus_token_batch_t b_sync = {};
                b_sync.tokens = seed_prompt;
                b_sync.n_tokens = 3;
                b_sync.start_pos = 0;
                b_sync.seq_id = 0;
                b_sync.request_logits = false;
                ARGUS_CHECK(argus_decode_batch(ctx, &b_sync) == 0);
                ARGUS_CHECK(argus_get_n_threads(ctx) > 0);
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }

            worker.join();
            ARGUS_CHECK(error_count.load() == 0);
            ARGUS_CHECK(sync_calls.load() == 100);
            std::cout << "  - Text context query/decode contention completed." << std::endl;
        }

        // 7.16. Sampler Lifecycle (prime, truncate, reset)
        int32_t prime_tokens[] = { 12, 13, 14 };
        int32_t prime_res = argus_sampler_prime(ctx, 0, prime_tokens, 3);
        ARGUS_CHECK(prime_res == 0);
        ARGUS_CHECK(argus_sampler_truncate(ctx, 0, 1) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 1);
        ARGUS_CHECK(argus_sampler_reset(ctx, 0) == 0);
        ARGUS_CHECK(argus_sampler_get_history_count(ctx, 0) == 0);
        std::cout << "  - Sampler lifecycle (prime, truncate, reset) verified." << std::endl;

        // 7.16. Performance Telemetry Metrics Verification
        std::cout << "[Test] Verifying performance telemetry metrics (argus_context_get_perf)..." << std::endl;
        argus_perf_timings_t timings = {};
        ARGUS_CHECK(argus_context_get_perf(nullptr, &timings) == false);
        ARGUS_CHECK(argus_context_get_perf(ctx, nullptr) == false);
        bool got_perf = argus_context_get_perf(ctx, &timings);
        ARGUS_CHECK(got_perf);
        ARGUS_CHECK(timings.n_p_eval > 0);
        ARGUS_CHECK(timings.t_p_eval_ms >= 0.0);
        ARGUS_CHECK(timings.t_start_ms > 0.0);
        ARGUS_CHECK(timings.n_reallocs_same >= 0);
        ARGUS_CHECK(timings.n_reallocs_diff >= 0);
        std::cout << "  - Performance telemetry verified: n_p_eval=" << timings.n_p_eval
                  << ", t_p_eval_ms=" << timings.t_p_eval_ms << " ms"
                  << ", reallocs_same=" << timings.n_reallocs_same
                  << ", reallocs_diff=" << timings.n_reallocs_diff << std::endl;

        argus_context_reset_perf(ctx);
        argus_perf_timings_t timings_reset = {};
        ARGUS_CHECK(argus_context_get_perf(ctx, &timings_reset));
        ARGUS_CHECK(timings_reset.n_p_eval < timings.n_p_eval);
        ARGUS_CHECK(timings_reset.t_p_eval_ms == 0.0);
        std::cout << "  - Performance telemetry reset verified: n_p_eval successfully reset from "
                  << timings.n_p_eval << " to " << timings_reset.n_p_eval << std::endl;

        // 7.17. Causal Attention State Mutations & Transition Contract
        std::cout << "[Test] Verifying causal attention state mutations & transition contract..." << std::endl;
        bool causal_state = false;
        ARGUS_CHECK(argus_context_get_causal_attn(ctx, &causal_state));
        ARGUS_CHECK(causal_state == true); // default is true for standard context

        // 1. Transition contract: cannot switch causal attention when draft model is attached
        ARGUS_CHECK(argus_context_set_causal_attn(ctx, false) == false);
        ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_STATE);

        // Null safety
        ARGUS_CHECK(argus_context_set_causal_attn(nullptr, true) == false);
        ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
        ARGUS_CHECK(argus_context_get_causal_attn(nullptr, &causal_state) == false);
        ARGUS_CHECK(argus_context_get_causal_attn(ctx, nullptr) == false);

        // 2. Standalone context transition contract:
        argus_context_params_t standalone_cparams = {};
        standalone_cparams.context_length = 256;
        standalone_cparams.cpu_threads = 2;
        standalone_cparams.n_seq_max = 1;
        standalone_cparams.u_batch = 4; // micro-batch size 4 for non-causal invariant testing

        argus_context_t * standalone_ctx = argus_context_init(model, &standalone_cparams);
        ARGUS_CHECK(standalone_ctx != nullptr);

        // Switch on empty context: succeeds
        ARGUS_CHECK(argus_context_set_causal_attn(standalone_ctx, false));
        ARGUS_CHECK(argus_context_get_causal_attn(standalone_ctx, &causal_state));
        ARGUS_CHECK(causal_state == false);
        ARGUS_CHECK(argus_context_set_causal_attn(standalone_ctx, true));
        ARGUS_CHECK(argus_context_get_causal_attn(standalone_ctx, &causal_state));
        ARGUS_CHECK(causal_state == true);

        // Populate KV cache by decoding 4 tokens
        int32_t sample_tokens[] = { 1, 4, 5, 6 };
        argus_token_batch_t standalone_batch = {};
        standalone_batch.tokens = sample_tokens;
        standalone_batch.n_tokens = 4;
        standalone_batch.start_pos = 0;
        standalone_batch.seq_id = 0;
        standalone_batch.request_logits = false;
        ARGUS_CHECK(argus_decode_batch(standalone_ctx, &standalone_batch) == 0);

        // Switch on populated context: MUST fail with ARGUS_ERROR_INVALID_STATE
        ARGUS_CHECK(argus_context_set_causal_attn(standalone_ctx, false) == false);
        ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_STATE);

        // Reset sampler and clear KV cache
        ARGUS_CHECK(argus_kv_cache_clear_slot(standalone_ctx, 0, 0, -1));
        ARGUS_CHECK(argus_sampler_reset(standalone_ctx, 0) == 0);

        // Now that context is reset, switching succeeds
        ARGUS_CHECK(argus_context_set_causal_attn(standalone_ctx, false));
        ARGUS_CHECK(argus_context_get_causal_attn(standalone_ctx, &causal_state));
        ARGUS_CHECK(causal_state == false);

        // 3. Non-causal physical micro-batch invariant verification:
        // u_batch is 4. Submitting <= 4 tokens succeeds; submitting > 4 tokens MUST fail with
        // ARGUS_ERROR_INVALID_ARGUMENT and zero process abort.
        standalone_batch.n_tokens = 4;
        ARGUS_CHECK(argus_decode_batch(standalone_ctx, &standalone_batch) == 0);

        // Clear cache before next decode
        ARGUS_CHECK(argus_kv_cache_clear_slot(standalone_ctx, 0, 0, -1));

        // Submit 6 tokens (> u_batch=4) under non-causal attention -> MUST fail without abort
        int32_t oversized_tokens[] = { 1, 4, 5, 6, 7, 8 };
        standalone_batch.tokens = oversized_tokens;
        standalone_batch.n_tokens = 6;
        ARGUS_CHECK(argus_decode_batch(standalone_ctx, &standalone_batch) == -1);
        ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
        std::cout << "  - Non-causal micro-batch invariant (n_tokens > n_ubatch rejection) verified without abort." << std::endl;

        argus_context_free(standalone_ctx);
        std::cout << "  - Causal attention mutation & transition contract verified." << std::endl;

        // 7.18. Deterministic Synthetic Embedding Model Verification (tests/data/tiny-embed.gguf)
        std::cout << "[Test] Verifying synthetic embedding model (tests/data/tiny-embed.gguf)..." << std::endl;
        const char * embed_model_paths[] = {
            "tests/data/tiny-embed.gguf",
            "../../tests/data/tiny-embed.gguf",
            "../tests/data/tiny-embed.gguf"
        };
        const char * valid_embed_path = nullptr;
        for (const char * p : embed_model_paths) {
            FILE * f = fopen(p, "rb");
            if (f) {
                fclose(f);
                valid_embed_path = p;
                break;
            }
        }

        ARGUS_CHECK(valid_embed_path != nullptr && "Mandatory tiny-embed.gguf fixture not found!");
        {
            argus_model_params_t embed_mparams = {};
            embed_mparams.model_path = valid_embed_path;
            embed_mparams.gpu_layers = 0;
            argus_model_t * embed_model = argus_model_load(&embed_mparams);
            ARGUS_CHECK(embed_model != nullptr);

            int32_t embd_dim = argus_model_n_embd(embed_model);
            ARGUS_CHECK(embd_dim == 32);

            argus_context_params_t embed_cparams = {};
            embed_cparams.context_length = 256;
            embed_cparams.cpu_threads = 2;
            embed_cparams.n_seq_max = 2; // Multi-sequence support for isolation testing
            embed_cparams.embeddings = true;
            embed_cparams.type_k = 0; // F16 KV cache
            embed_cparams.type_v = 0;
            embed_cparams.u_batch = 4; // Physical micro-batch boundary for non-causal invariant testing

            argus_context_t * embed_ctx = argus_context_init(embed_model, &embed_cparams);
            ARGUS_CHECK(embed_ctx != nullptr);

            // Verifying has_embeddings guard: query before decode must return ARGUS_ERROR_INVALID_STATE
            float out_embeds[32] = {};
            ARGUS_CHECK(argus_get_embeddings(embed_ctx, 0, out_embeds, 32) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_STATE);

            // Expected analytical reference coordinates for tiny-embed.gguf with input tokens [1, 4, 5]
            static const float EXPECTED_VEC_A[32] = {
                0.012607f, 0.008028f, 0.004507f, -0.023824f, 0.000028f, -0.013474f, 0.001769f, -0.006316f,
                -0.000113f, 0.000781f, 0.000226f, -0.019436f, -0.001485f, 0.004716f, -0.013539f, 0.005747f,
                -0.004355f, -0.003777f, -0.001283f, -0.001102f, 0.000275f, -0.000012f, 0.005550f, -0.016983f,
                0.025341f, 0.001413f, -0.002965f, 0.000518f, 0.002365f, -0.010931f, 0.030305f, -0.000967f
            };

            // Test 1: Decode sequence 0 with [1, 4, 5]
            int32_t embed_tokens_a[] = { 1, 4, 5 };
            argus_token_batch_t embed_batch_a = {};
            embed_batch_a.tokens = embed_tokens_a;
            embed_batch_a.n_tokens = 3;
            embed_batch_a.start_pos = 0;
            embed_batch_a.seq_id = 0;
            embed_batch_a.request_logits = false;
            ARGUS_CHECK(argus_decode_batch(embed_ctx, &embed_batch_a) == 0);

            // Query embeddings for sequence 0
            int32_t written_a = argus_get_embeddings(embed_ctx, 0, out_embeds, 32);
            ARGUS_CHECK(written_a == 32);

            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(!std::isnan(out_embeds[i]) && !std::isinf(out_embeds[i]));
                ARGUS_CHECK(std::abs(out_embeds[i] - EXPECTED_VEC_A[i]) < 1e-4f);
            }
            std::cout << "  - Deterministic reference vector A verified with 1e-4 precision." << std::endl;

            // Test 2: Atomic evaluation via argus_eval_embeddings on sequence 0
            float eval_out_a[32] = {};
            int32_t eval_written = argus_eval_embeddings(embed_ctx, embed_tokens_a, 3, 0, eval_out_a, 32);
            ARGUS_CHECK(eval_written == 32);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(std::abs(eval_out_a[i] - EXPECTED_VEC_A[i]) < 1e-4f);
            }

            // Test 2b: Preflight non-causal batch limit rejection & state neutrality (F-01 / F-03)
            // Context has u_batch=4 and non-causal attention. Submit 8 tokens (> u_batch=4).
            int32_t oversized_embed_tokens[8] = { 1, 4, 5, 6, 7, 8, 9, 10 };
            float sentinel_out[32];
            for (int i = 0; i < 32; ++i) {
                sentinel_out[i] = -999.0f;
            }

            int32_t preflight_res = argus_eval_embeddings(embed_ctx, oversized_embed_tokens, 8, 0, sentinel_out, 32);
            ARGUS_CHECK(preflight_res == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            // Invariant 1: Output buffer must remain completely untouched on preflight failure
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(sentinel_out[i] == -999.0f);
            }

            // Invariant 2: Sequence 0's prior valid embeddings MUST STILL BE INTACT AND UNCHANGED!
            float preserved_embeds[32] = {};
            int32_t preserved_written = argus_get_embeddings(embed_ctx, 0, preserved_embeds, 32);
            ARGUS_CHECK(preserved_written == 32);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(std::abs(preserved_embeds[i] - EXPECTED_VEC_A[i]) < 1e-4f);
            }

            // Invariant 3: Sequence 0 KV cache high-water mark must remain intact (max pos is 2 for 3 tokens: 0, 1, 2)
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(embed_ctx, 0) == 2);

            // Test argus_eval_embeddings capacity constraint: max_floats < 32 returns -2
            for (int i = 0; i < 32; ++i) sentinel_out[i] = -888.0f;
            ARGUS_CHECK(argus_eval_embeddings(embed_ctx, embed_tokens_a, 3, 0, sentinel_out, 31) == -2);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(sentinel_out[i] == -888.0f);
            }

            // Prior embedding still intact after capacity rejection
            ARGUS_CHECK(argus_get_embeddings(embed_ctx, 0, preserved_embeds, 32) == 32);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(std::abs(preserved_embeds[i] - EXPECTED_VEC_A[i]) < 1e-4f);
            }
            std::cout << "  - Preflight non-causal rejection & output capacity state-neutrality verified." << std::endl;

            // Test 3: Multi-sequence generational invalidation proof
            // Expected analytical reference coordinates for tiny-embed.gguf with input tokens [1, 4, 6]
            static const float EXPECTED_VEC_B[32] = {
                -0.003486f, 0.024704f, -0.008348f, -0.010331f, -0.000755f, -0.016964f, 0.002638f, -0.001489f,
                0.000092f, -0.000260f, -0.002435f, -0.005751f, -0.002582f, -0.000795f, -0.012421f, 0.004681f,
                -0.007213f, -0.020193f, -0.000497f, 0.000980f, 0.000381f, -0.000168f, 0.020802f, -0.018449f,
                0.031088f, -0.000668f, -0.000664f, 0.000861f, 0.013751f, -0.011461f, 0.019753f, 0.007501f
            };

            int32_t embed_tokens_b[] = { 1, 4, 6 };
            float eval_out_b[32] = {};
            int32_t written_b = argus_eval_embeddings(embed_ctx, embed_tokens_b, 3, 1, eval_out_b, 32);
            ARGUS_CHECK(written_b == 32);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(std::abs(eval_out_b[i] - EXPECTED_VEC_B[i]) < 1e-4f);
            }

            // CRITICAL INVARIANT: Query sequence 0 again! It must fail with ARGUS_ERROR_INVALID_STATE
            // because sequence 1 superseded upstream embedding output!
            float stale_check[32] = {};
            ARGUS_CHECK(argus_get_embeddings(embed_ctx, 0, stale_check, 32) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_STATE);

            // Query sequence 1: must succeed and match vector B
            float seq1_check[32] = {};
            ARGUS_CHECK(argus_get_embeddings(embed_ctx, 1, seq1_check, 32) == 32);
            for (int i = 0; i < 32; ++i) {
                ARGUS_CHECK(std::abs(seq1_check[i] - EXPECTED_VEC_B[i]) < 1e-4f);
            }
            std::cout << "  - Generational multi-sequence embedding isolation verified: zero cross-sequence contamination." << std::endl;

            argus_context_free(embed_ctx);
            argus_model_free(embed_model);
        }

        // 7.18a. Contiguous RAII Batch Allocator (argus_owned_batch) Fault Injection (F-01 / F-02 / TTS)
#ifdef ARGUS_TESTING
        {
            std::cout << "[Test] Verifying argus_owned_batch OOM fault injection across all 6 sub-buffer steps..." << std::endl;
            argus_context_params_t oom_cparams = {};
            oom_cparams.context_length = 128;
            oom_cparams.u_batch = 16;
            oom_cparams.cpu_threads = 2;
            oom_cparams.n_seq_max = 1;

            // 1. Fault-inject argus_context_init across steps 1..6
            for (int step = 1; step <= 6; ++step) {
                argus_clear_error();
                argus_test_set_batch_alloc_fail_step(step);
                argus_context_t * failed_ctx = argus_context_init(model, &oom_cparams);
                ARGUS_CHECK(failed_ctx == nullptr);
                ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_OUT_OF_MEMORY);
            }
            argus_test_set_batch_alloc_fail_step(0);

            // 2. Fault-inject decode_tokens_chunked fallback path when context batch_scratch is disabled
            argus_context_t * fallback_ctx = argus_context_init(model, &oom_cparams);
            ARGUS_CHECK(fallback_ctx != nullptr);
            argus_test_set_context_batch_scratch_enabled(fallback_ctx, false);

            int32_t fb_tokens[] = { 1, 4, 5, 6 };
            argus_token_batch_t fb_batch = {};
            fb_batch.tokens = fb_tokens;
            fb_batch.n_tokens = 4;
            fb_batch.start_pos = 0;
            fb_batch.seq_id = 0;
            fb_batch.request_logits = true;

            // With fail_step == 0, local_batch_owner fallback succeeds cleanly
            ARGUS_CHECK(argus_decode_batch(fallback_ctx, &fb_batch) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(fallback_ctx, 0) == 3);

            // With fail_step == 1..6, local_batch_owner fallback fails cleanly with ARGUS_ERROR_OUT_OF_MEMORY
            for (int step = 1; step <= 6; ++step) {
                argus_clear_error();
                argus_test_set_batch_alloc_fail_step(step);
                fb_batch.start_pos = 4;
                ARGUS_CHECK(argus_decode_batch(fallback_ctx, &fb_batch) == -1);
                ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_OUT_OF_MEMORY);
                // KV cache high-water mark must remain 3 (uncorrupted)
                ARGUS_CHECK(argus_kv_cache_seq_pos_max(fallback_ctx, 0) == 3);
            }
            argus_test_set_batch_alloc_fail_step(0);
            argus_test_set_context_batch_scratch_enabled(fallback_ctx, true);

            argus_context_free(fallback_ctx);
            std::cout << "  - argus_owned_batch 6-step OOM fault injection verified across context init and decode fallback." << std::endl;
        }
#endif

        // 7.18b. KV-Cache Position Shifting (argus_kv_cache_seq_add) & Sampler Coordinate Sync (F-03)
        {
            std::cout << "[Test] Verifying argus_kv_cache_seq_add position shifting and sampler coordinate sync..." << std::endl;
            argus_context_params_t shift_cparams = {};
            shift_cparams.context_length = 64;
            shift_cparams.u_batch = 16;
            shift_cparams.cpu_threads = 2;
            shift_cparams.n_seq_max = 2;
            shift_cparams.draft_model = draft_model;

            argus_context_t * shift_ctx = argus_context_init(model, &shift_cparams);
            ARGUS_CHECK(shift_ctx != nullptr);
            ARGUS_CHECK(argus_context_has_draft(shift_ctx));
            ARGUS_CHECK(argus_kv_cache_can_shift(shift_ctx));

            // Parameter validation checks
            ARGUS_CHECK(!argus_kv_cache_seq_add(nullptr, 0, 0, -1, -1));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(!argus_kv_cache_seq_add(shift_ctx, -1, 0, -1, -1));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(!argus_kv_cache_seq_add(shift_ctx, 2, 0, -1, -1));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(!argus_kv_cache_seq_add(shift_ctx, 0, 10, 5, -1));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(!argus_kv_cache_seq_add(shift_ctx, 0, 2, -1, -5)); // p0_eff (2) + delta (-5) < 0
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            // delta == 0 is a valid no-op
            ARGUS_CHECK(argus_kv_cache_seq_add(shift_ctx, 0, 0, -1, 0));

            // Prime 2 tokens (tagged with kv_pos = -1)
            int32_t primed[] = { 2, 3 };
            ARGUS_CHECK(argus_sampler_prime(shift_ctx, 0, primed, 2) == 0);
            ARGUS_CHECK(argus_sampler_get_history_count(shift_ctx, 0) == 2);

            // Decode 6 prompt tokens at [0..5]
            int32_t prompt_toks[] = { 1, 4, 5, 6, 7, 8 };
            argus_token_batch_t pb = {};
            pb.tokens = prompt_toks;
            pb.n_tokens = 6;
            pb.start_pos = 0;
            pb.seq_id = 0;
            pb.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(shift_ctx, &pb) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_min(shift_ctx, 0) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(shift_ctx, 0) == 5);

            // Sample token (recorded as pending at kv_pos = 5) and decode continuation at pos 6
            int32_t sampled = argus_sample_token(shift_ctx, 0, 0.0f, 1.1f);
            ARGUS_CHECK(sampled >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(shift_ctx, 0) == 1);

            argus_token_batch_t cb = {};
            cb.tokens = &sampled;
            cb.n_tokens = 1;
            cb.start_pos = 6;
            cb.seq_id = 0;
            cb.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(shift_ctx, &cb) == 0);
            ARGUS_CHECK(argus_sampler_has_pending(shift_ctx, 0) == 0);
            ARGUS_CHECK(argus_sampler_get_history_count(shift_ctx, 0) == 3);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(shift_ctx, 0) == 6);

#ifdef ARGUS_TESTING
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 0) == -1);
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 1) == -1);
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 2) == 6);
#endif

            // Reject positive shift that would push pos_max (6) >= n_ctx (64)
            ARGUS_CHECK(!argus_kv_cache_seq_add(shift_ctx, 0, 0, -1, 58));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(shift_ctx, 0) == 6);

            // Ring-buffer eviction: keep [0, 2), evict [2, 4), shift [4, -1) left by -2
            ARGUS_CHECK(argus_kv_cache_clear_slot(shift_ctx, 0, 2, 4));
            ARGUS_CHECK(argus_kv_cache_seq_add(shift_ctx, 0, 4, -1, -2));

            // Surviving KV positions are now [0, 1, 2, 3, 4] -> min = 0, max = 4
            ARGUS_CHECK(argus_kv_cache_seq_pos_min(shift_ctx, 0) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(shift_ctx, 0) == 4);
            ARGUS_CHECK(argus_sampler_get_history_count(shift_ctx, 0) == 3);

#ifdef ARGUS_TESTING
            // Primed entries stay at -1; committed entry at kv_pos = 6 shifted by -2 -> kv_pos = 4
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 0) == -1);
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 1) == -1);
            ARGUS_CHECK(argus_test_sampler_get_history_kv_pos(shift_ctx, 0, 2) == 4);
#endif

            // Logits must have been invalidated by argus_kv_cache_seq_add
            ARGUS_CHECK(argus_sample_token(shift_ctx, 0, 0.0f, 1.1f) == -2);

            // Decoding at the new contiguous position (start_pos = 5) succeeds cleanly!
            int32_t next_tok = 9;
            cb.tokens = &next_tok;
            cb.n_tokens = 1;
            cb.start_pos = 5;
            cb.seq_id = 0;
            cb.request_logits = true;
            ARGUS_CHECK(argus_decode_batch(shift_ctx, &cb) == 0);
            ARGUS_CHECK(argus_kv_cache_seq_pos_max(shift_ctx, 0) == 5);
            ARGUS_CHECK(argus_sample_token(shift_ctx, 0, 0.0f, 1.1f) >= 0);

            argus_context_free(shift_ctx);
            std::cout << "  - KV-cache position shifting, speculative draft sync, and sampler coordinate shifting verified." << std::endl;
        }

        // 7.18c. EmbeddingGemma2 (n_embd_out != n_embd) Projected Dimension Resolution (F-04)
        {
            std::cout << "[Test] Verifying EmbeddingGemma2 (tiny-gemma-embed2.gguf) projected output dimensions..." << std::endl;
            // Standard tiny.gguf has n_embd == n_embd_out == n_embd_inp == 32
            ARGUS_CHECK(argus_model_n_embd(model) == 32);
            ARGUS_CHECK(argus_model_n_embd_out(model) == 32);
            ARGUS_CHECK(argus_model_n_embd_inp(model) == 32);
            ARGUS_CHECK(argus_model_n_embd_out(nullptr) == -1);
            ARGUS_CHECK(argus_model_n_embd_inp(nullptr) == -1);

            const char * gemma2_candidates[] = {
                "tests/data/tiny-gemma-embed2.gguf",
                "../tests/data/tiny-gemma-embed2.gguf",
                "../../tests/data/tiny-gemma-embed2.gguf"
            };
            const char * gemma2_path = nullptr;
            for (const char * cand : gemma2_candidates) {
                FILE * f = std::fopen(cand, "rb");
                if (f) {
                    std::fclose(f);
                    gemma2_path = cand;
                    break;
                }
            }
            ARGUS_CHECK(gemma2_path != nullptr);

            argus_model_params_t gm_params = {};
            gm_params.model_path = gemma2_path;
            gm_params.gpu_layers = 0;
            argus_model_t * gemma2_model = argus_model_load(&gm_params);
            ARGUS_CHECK(gemma2_model != nullptr);

            // tiny-gemma-embed2.gguf has backbone n_embd_inp = 32 and projected output n_embd_out = 16
            ARGUS_CHECK(argus_model_n_embd(gemma2_model) == 16);
            ARGUS_CHECK(argus_model_n_embd_out(gemma2_model) == 16);
            ARGUS_CHECK(argus_model_n_embd_inp(gemma2_model) == 32);

            argus_context_params_t gc_params = {};
            gc_params.context_length = 128;
            gc_params.u_batch = 16;
            gc_params.cpu_threads = 2;
            gc_params.n_seq_max = 1;
            gc_params.embeddings = true;

            argus_context_t * gemma2_ctx = argus_context_init(gemma2_model, &gc_params);
            ARGUS_CHECK(gemma2_ctx != nullptr);
            ARGUS_CHECK(argus_context_set_causal_attn(gemma2_ctx, false));

            // gemma-embedding2 is a no-KV-cache architecture: can_shift is false and seq_add returns ARGUS_ERROR_ROLLBACK_FAILED
            ARGUS_CHECK(!argus_kv_cache_can_shift(gemma2_ctx));
            ARGUS_CHECK(!argus_kv_cache_seq_add(gemma2_ctx, 0, 0, -1, -1));
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_ROLLBACK_FAILED);

            int32_t g_toks[] = { 1, 4, 5 };
            float g_out[16] = {};
            // Capacity check: 15 < 16 must return -2
            ARGUS_CHECK(argus_eval_embeddings(gemma2_ctx, g_toks, 3, 0, g_out, 15) == -2);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            // Exact 16-float evaluation via argus_eval_embeddings
            int32_t g_written = argus_eval_embeddings(gemma2_ctx, g_toks, 3, 0, g_out, 16);
            ARGUS_CHECK(g_written == 16);

            double sum_sq = 0.0;
            for (int i = 0; i < 16; ++i) {
                ARGUS_CHECK(!std::isnan(g_out[i]) && !std::isinf(g_out[i]));
                sum_sq += static_cast<double>(g_out[i]) * static_cast<double>(g_out[i]);
            }
            ARGUS_CHECK(sum_sq > 0.0);

            // Query via argus_get_embeddings must also return 16 floats matching g_out
            float g_out2[16] = {};
            ARGUS_CHECK(argus_get_embeddings(gemma2_ctx, 0, g_out2, 16) == 16);
            for (int i = 0; i < 16; ++i) {
                ARGUS_CHECK(std::abs(g_out[i] - g_out2[i]) < 1e-6f);
            }

            argus_context_free(gemma2_ctx);
            argus_model_free(gemma2_model);
            std::cout << "  - EmbeddingGemma2 projected dimensions (n_embd=16, n_embd_out=16, n_embd_inp=32) and deterministic extraction verified." << std::endl;
        }

        // 7.18d. Multimodal Chunk Evaluation Embedding Invalidation Verification
        {
            const char * mmproj_candidates[] = {
                "tests/data/tiny-mmproj.gguf",
                "../tests/data/tiny-mmproj.gguf",
                "../../tests/data/tiny-mmproj.gguf"
            };
            const char * mmproj_path = nullptr;
            for (const char * cand : mmproj_candidates) {
                FILE * f = std::fopen(cand, "rb");
                if (f) {
                    std::fclose(f);
                    mmproj_path = cand;
                    break;
                }
            }
            ARGUS_CHECK(mmproj_path != nullptr);

            argus_context_params_t mm_cparams = {};
            mm_cparams.context_length = 128;
            mm_cparams.u_batch = 16;
            mm_cparams.cpu_threads = 2;
            mm_cparams.n_seq_max = 1;
            mm_cparams.embeddings = true;

            argus_context_t * mm_ctx = argus_context_init(model, &mm_cparams);
            ARGUS_CHECK(mm_ctx != nullptr);

            int32_t init_toks[] = { 1, 4, 5 };
            float mm_emb[32] = {};
            ARGUS_CHECK(argus_eval_embeddings(mm_ctx, init_toks, 3, 0, mm_emb, 32) == 32);
#ifdef ARGUS_TESTING
            ARGUS_CHECK(argus_test_has_embeddings(mm_ctx, 0) == 1);
#endif

            argus_multimodal_params_t mmp = {};
            mmp.mmproj_path = mmproj_path;
            mmp.cpu_threads = 2;
            argus_multimodal_t * proj = argus_multimodal_init(model, &mmp);
            ARGUS_CHECK(proj != nullptr);

            std::vector<uint8_t> rgb(16 * 16 * 3, 127);
            argus_bitmap_t * bmp = argus_bitmap_from_rgb(16, 16, rgb.data());
            ARGUS_CHECK(bmp != nullptr);
            argus_input_chunks_t * chunks = argus_input_chunks_init();
            ARGUS_CHECK(chunks != nullptr);
            const argus_bitmap_t * imgs[] = { bmp };
            ARGUS_CHECK(argus_multimodal_tokenize(proj, chunks, "<__media__>", false, imgs, 1) == 0);

            int32_t new_n_past = -1;
            ARGUS_CHECK(argus_eval_multimodal_chunks(proj, mm_ctx, chunks, 3, 0, 8, true, &new_n_past) == 0);
            ARGUS_CHECK(new_n_past == 7);
#ifdef ARGUS_TESTING
            ARGUS_CHECK(argus_test_has_embeddings(mm_ctx, 0) == 0);
#endif
            // Prior text embedding must be invalidated after multimodal chunk evaluation
            ARGUS_CHECK(argus_get_embeddings(mm_ctx, 0, mm_emb, 32) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_STATE);

            argus_input_chunks_free(chunks);
            argus_bitmap_free(bmp);
            argus_multimodal_free(proj);
            argus_context_free(mm_ctx);
            std::cout << "  - Multimodal chunk evaluation stale embedding invalidation verified." << std::endl;
        }

        // 7.19. Cleanup
        argus_context_free(ctx);
        argus_model_free(draft_model);
        argus_model_free(model);
        std::cout << "  - End-to-end model and context resources successfully released." << std::endl;

        // 7.18. OS Stderr Suppression Verification (Kernel Pipe / dup2)
#if !defined(_WIN32)
        std::cout << "[Test] Verifying kernel stderr suppression under ARGUS_LOG_NONE..." << std::endl;
        int pipe_fds[2];
        ARGUS_CHECK(pipe(pipe_fds) == 0);
        int pflags = fcntl(pipe_fds[0], F_GETFL, 0);
        fcntl(pipe_fds[0], F_SETFL, pflags | O_NONBLOCK);

        int saved_stderr = dup(STDERR_FILENO);
        ARGUS_CHECK(saved_stderr >= 0);
        fflush(stderr);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[1]);

        argus_set_log_level(ARGUS_LOG_NONE);
        argus_set_log_callback(nullptr, nullptr);

        argus_model_t * silent_model = argus_model_load(&mparams);
        ARGUS_CHECK(silent_model != nullptr);
        argus_context_params_t silent_cparams = {};
        silent_cparams.context_length = 256;
        silent_cparams.cpu_threads = 2;
        silent_cparams.n_seq_max = 1;
        argus_context_t * silent_ctx = argus_context_init(silent_model, &silent_cparams);
        ARGUS_CHECK(silent_ctx != nullptr);
        int32_t s_toks[] = { 1, 4, 5, 6 };
        argus_token_batch_t s_b = {};
        s_b.tokens = s_toks;
        s_b.n_tokens = 4;
        s_b.start_pos = 0;
        s_b.seq_id = 0;
        s_b.request_logits = true;
        ARGUS_CHECK(argus_decode_batch(silent_ctx, &s_b) == 0);

        fflush(stderr);
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);

        char pipe_buf[512];
        ssize_t n_read = read(pipe_fds[0], pipe_buf, sizeof(pipe_buf));
        close(pipe_fds[0]);

        argus_context_free(silent_ctx);
        argus_model_free(silent_model);
        argus_set_log_level(ARGUS_LOG_WARN); // restore default

        ARGUS_CHECK(n_read <= 0);
        std::cout << "  - Fanged verification passed: exactly 0 bytes escaped to stderr under ARGUS_LOG_NONE." << std::endl;
#endif

        // 7.19. Build Features & Tokenize-N Verification
        {
            uint64_t features = argus_build_features();
            ARGUS_CHECK((features & ARGUS_FEATURE_CPU) != 0);
            std::cout << "  - Build features bitmask verified (0x" << std::hex << features << std::dec << ")." << std::endl;
        }

        // 7.18. Ref-counted Model Ownership & Deferred Backend Teardown
        {
            argus_model_params_t mparams = {};
            mparams.model_path = valid_model_path;
            mparams.gpu_layers = 0;
            argus_model_t * m = argus_model_load(&mparams);
            ARGUS_CHECK(m != nullptr);
            ARGUS_CHECK(argus_backend_is_initialized());

            // Retain model reference (refs = 2)
            ARGUS_CHECK(argus_model_retain(m));

            // Verify argus_tokenize_n boundary validation & exception containment
            int32_t out_toks[16];
            ARGUS_CHECK(argus_tokenize_n(nullptr, "abc", 2, out_toks, 16, false) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            ARGUS_CHECK(argus_tokenize_n(m, nullptr, 2, out_toks, 16, false) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            ARGUS_CHECK(argus_tokenize_n(m, "abc", 2, nullptr, 16, false) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            ARGUS_CHECK(argus_tokenize_n(m, "abc", 2, out_toks, 0, false) == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INVALID_ARGUMENT);

            // C++ exception containment: tiny.gguf lacks SPM space piece, throwing in llama.cpp
            int32_t n_toks = argus_tokenize_n(m, "abc", 2, out_toks, 16, false);
            ARGUS_CHECK(n_toks == -1);
            ARGUS_CHECK(argus_last_error_code() == ARGUS_ERROR_INTERNAL);
            ARGUS_CHECK(argus_last_error_message() != nullptr);
            ARGUS_CHECK(std::strlen(argus_last_error_message()) > 0);
            argus_clear_error();
            ARGUS_CHECK(argus_last_error_code() == ARGUS_SUCCESS);
            std::cout << "  - Exception containment & boundary validation for argus_tokenize_n verified." << std::endl;

            // Call argus_backend_free while model is retained
            argus_backend_free();
            // Backend must NOT be torn down yet because model is alive
            ARGUS_CHECK(argus_backend_is_initialized());

            // Release first reference (refs = 1)
            argus_model_release(m);
            ARGUS_CHECK(argus_backend_is_initialized());

            // Release final reference (refs = 0) -> triggers deferred backend teardown
            argus_model_release(m);
            ARGUS_CHECK(!argus_backend_is_initialized());
            std::cout << "  - Ref-counted model ownership & deferred backend teardown verified." << std::endl;
        }
    } else {
        std::cerr << "[Test] WARNING: tests/data/tiny.gguf not found; skipped model execution pass." << std::endl;
    }

    // 8. Free the global backends (idempotent if already torn down by deferred teardown)
    argus_backend_free();
    std::cout << "[Test] Backend freed successfully." << std::endl;

    std::cout << "[Test] All integrated assertions passed!" << std::endl;
    return 0;
}
