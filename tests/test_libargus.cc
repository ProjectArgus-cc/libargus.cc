/**
 * @file test_libargus.cc
 * @brief Comprehensive integration and fanged unit tests for libargus lifecycle, backend telemetry,
 *        and state-machine sampling verification.
 */

#include "libargus.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>
#include <set>
#include <thread>
#include <chrono>
#include <atomic>

#define ARGUS_CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

int main() {
    std::cout << "[Test] Starting libargus lifecycle integration verification..." << std::endl;

    // 1. Initialize the global backends
    bool init_success = argus_backend_init(nullptr);
    ARGUS_CHECK(init_success);
    std::cout << "[Test] Backend initialized successfully." << std::endl;

    // Assert compiled version query matches expectations
    std::cout << "[Test] Library Version: " << argus_version() << std::endl;
    ARGUS_CHECK(std::strcmp(argus_version(), "1.7.0") == 0);

    // 2. Query backend count and list their names
    int32_t backend_count = argus_backend_get_count();
    std::cout << "[Test] Detected " << backend_count << " available hardware backend(s):" << std::endl;

    for (int32_t i = 0; i < backend_count; ++i) {
        const char * backend_name = argus_backend_get_name(i);
        std::cout << "  - Backend " << i << ": " << (backend_name ? backend_name : "UNKNOWN") << std::endl;
    }

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
        argus_kv_cache_clear_slot(ctx, 0, 1, -1);
        int32_t mutated_sample = argus_sample_token_ext(ctx, 0, &sparams, nullptr, 0);
        ARGUS_CHECK(mutated_sample == -2);
        std::cout << "  - Stale logits invalidation verified: KV cache mutation invalidated pending logits (-2)." << std::endl;

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

            // Multimodal evaluation automatic pending discard
            ARGUS_CHECK(argus_decode_batch(ctx, &b_n) == 0);
            int32_t pend_mm = argus_sample_token_ext(ctx, 0, &seed_sparams, nullptr, 0);
            ARGUS_CHECK(pend_mm >= 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 1);
            ARGUS_CHECK(argus_multimodal_test_lock_sync(ctx, 0, 0) == 0);
            ARGUS_CHECK(argus_sampler_has_pending(ctx, 0) == 0);
            std::cout << "  - Ghost token elimination & canonical pending discard verified." << std::endl;
        }

        // 7.15. Multimodal Context Mutex Serialization & Concurrency
        {
            std::atomic<bool> worker_done{false};
            std::atomic<int32_t> error_count{0};
            std::atomic<int32_t> sync_calls{0};

            std::thread worker([&]() {
                for (int iter = 0; iter < 100; ++iter) {
                    // Actively acquires and holds ctx->mtx for 100 microseconds per iteration
                    int32_t res = argus_multimodal_test_lock_sync(ctx, 0, 100);
                    if (res != 0) {
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
            std::cout << "  - Multimodal context mutex serialization verified under active contention (100 synchronized passes)." << std::endl;
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

        // 7.15. Cleanup
        argus_context_free(ctx);
        argus_model_free(draft_model);
        argus_model_free(model);
        std::cout << "  - End-to-end model and context resources successfully released." << std::endl;

        // 7.17. Build Features & Tokenize-N Verification
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
            ARGUS_CHECK(std::strstr(argus_last_error_message(), "unordered_map::at") != nullptr);
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
