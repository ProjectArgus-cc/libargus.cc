/**
 * @file test_libargus.cc
 * @brief Simple integration test for libargus lifecycle and backend telemetry.
 */

#include "libargus.h"
#include <iostream>
#include <cassert>
#include <string>
#include <cstring>

int main() {
    std::cout << "[Test] Starting libargus lifecycle integration verification..." << std::endl;

    // 1. Initialize the global backends
    bool init_success = argus_backend_init(nullptr);
    if (!init_success) {
        std::cerr << "[Test] Failed to initialize argus backends!" << std::endl;
        return 1;
    }
    std::cout << "[Test] Backend initialized successfully." << std::endl;

    // Assert compiled version query matches expectations
    std::cout << "[Test] Library Version: " << argus_version() << std::endl;
    assert(std::strcmp(argus_version(), "1.6.0") == 0);

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
    assert(null_model == nullptr);

    // 4. Test audio context initialization with null input
    std::cout << "[Test] Verifying audio context init handling with null input..." << std::endl;
    argus_audio_context_t * null_audio_ctx = argus_audio_init(nullptr);
    assert(null_audio_ctx == nullptr);

    // 5. Test TTS output generation on a placeholder context
    std::cout << "[Test] Verifying speech synthesis dummy float generator..." << std::endl;
    float dummy_pcm[100];
    // Passing null context should return -1
    int32_t samples = argus_synthesize_speech(nullptr, nullptr, "Hello test", 42, dummy_pcm, 100, nullptr, 0);
    assert(samples == -1);

    // 6. Test Multimodal initialization and load boundaries with invalid/null params
    std::cout << "[Test] Verifying multimodal init handling with null input..." << std::endl;
    argus_multimodal_t * null_mctx = argus_multimodal_init(nullptr, nullptr);
    assert(null_mctx == nullptr);

    std::cout << "[Test] Verifying support queries on null multimodal context..." << std::endl;
    assert(argus_multimodal_support_vision(nullptr) == false);
    assert(argus_multimodal_support_audio(nullptr) == false);
    assert(argus_multimodal_support_video(nullptr) == false);
    assert(argus_multimodal_get_audio_sample_rate(nullptr) == -1);

    std::cout << "[Test] Verifying bitmap creation on null inputs..." << std::endl;
    assert(argus_bitmap_from_rgb(100, 100, nullptr) == nullptr);
    assert(argus_bitmap_from_pcm(nullptr, 100) == nullptr);
    assert(argus_bitmap_load_file(nullptr, nullptr, false) == nullptr);
    assert(argus_bitmap_load_buffer(nullptr, nullptr, 0, false) == nullptr);

    std::cout << "[Test] Verifying video loaders and reader boundary handling..." << std::endl;
    assert(argus_video_load_file(nullptr, nullptr, 0.0f, 0) == nullptr);
    assert(argus_video_load_buffer(nullptr, nullptr, 0, 0.0f, 0) == nullptr);
    assert(argus_video_read_next(nullptr, nullptr, nullptr, 0) == -2);

    std::cout << "[Test] Verifying chunks allocation and tokenization boundaries..." << std::endl;
    argus_input_chunks_t * chunks = argus_input_chunks_init();
    assert(chunks != nullptr);

    // Tokenization with null context or invalid bitmaps array should return -1
    int32_t tok_res = argus_multimodal_tokenize(nullptr, chunks, "test", true, nullptr, 0);
    assert(tok_res == -1);

    // Evaluation with null contexts should return -1
    int32_t out_n_past = 0;
    int32_t eval_res = argus_eval_multimodal_chunks(nullptr, nullptr, chunks, 0, 0, 128, false, &out_n_past);
    assert(eval_res == -1);

    argus_input_chunks_free(chunks);
    std::cout << "[Test] Multimodal C API tests completed successfully." << std::endl;

    // 6.5. Test new Vocabulary and Metadata null checks
    std::cout << "[Test] Verifying vocab and metadata null safety checks..." << std::endl;
    assert(argus_vocab_bos(nullptr) == -1);
    assert(argus_vocab_eos(nullptr) == -1);
    assert(argus_vocab_eot(nullptr) == -1);
    assert(argus_vocab_pad(nullptr) == -1);
    assert(argus_vocab_n_tokens(nullptr) == -1);
    assert(argus_vocab_is_eog(nullptr, 0) == false);

    char dummy_meta_buf[10];
    assert(argus_model_meta_val_str(nullptr, "some_key", dummy_meta_buf, 10) == -1);
    assert(argus_model_meta_count(nullptr) == -1);
    assert(argus_model_meta_key_by_index(nullptr, 0, dummy_meta_buf, 10) == -1);
    assert(argus_model_meta_val_str_by_index(nullptr, 0, dummy_meta_buf, 10) == -1);

    // Test new Model Shape queries null safety checks
    assert(argus_model_n_embd(nullptr) == -1);
    assert(argus_model_n_ctx_train(nullptr) == -1);
    assert(argus_model_n_layer(nullptr) == -1);
    assert(argus_model_n_head(nullptr) == -1);
    assert(argus_model_n_head_kv(nullptr) == -1);
    assert(argus_model_n_params(nullptr) == 0);
    assert(argus_model_has_encoder(nullptr) == false);
    assert(argus_model_n_pos_per_embd(nullptr) == -1);
    assert(argus_model_is_mrope(nullptr) == false);

    // Test KV cache position queries null safety checks
    assert(argus_kv_cache_seq_pos_max(nullptr, 0) == -1);
    assert(argus_kv_cache_seq_pos_min(nullptr, 0) == -1);

    // Test new Model KV memory calculation, size, desc, and GGML quantization introspection null safety
    std::cout << "[Test] Verifying KV cache calculation and quantization introspection null safety..." << std::endl;
    assert(argus_model_size(nullptr) == 0);
    assert(argus_model_desc(nullptr, dummy_meta_buf, 10) == -1);
    assert(argus_model_desc(nullptr, nullptr, 0) == -1);
    assert(argus_model_kv_bytes_per_token(nullptr, ARGUS_KV_TYPE_F16, ARGUS_KV_TYPE_F16) == -1);
    assert(argus_model_estimate_vram_bytes(nullptr, 4096, ARGUS_KV_TYPE_Q4_0, ARGUS_KV_TYPE_Q4_0) == -1);
    assert(argus_model_estimate_vram_bytes(nullptr, -10, ARGUS_KV_TYPE_Q4_0, ARGUS_KV_TYPE_Q4_0) == -1);

    // Test GGML quantization type size and block size calculation logic
    std::cout << "[Test] Verifying GGML quantization type and block size calculations..." << std::endl;
    assert(argus_quant_type_size(ARGUS_KV_TYPE_F16) == 2);
    assert(argus_quant_block_size(ARGUS_KV_TYPE_F16) == 1);
    assert(argus_quant_type_size(ARGUS_KV_TYPE_Q8_0) > 0);
    assert(argus_quant_block_size(ARGUS_KV_TYPE_Q8_0) > 0);
    assert(argus_quant_type_size(ARGUS_KV_TYPE_Q4_0) > 0);
    assert(argus_quant_block_size(ARGUS_KV_TYPE_Q4_0) > 0);
    assert(argus_quant_type_size(-1) == 0);
    assert(argus_quant_block_size(-1) == 0);
    assert(argus_quant_type_size(99999) == 0);
    assert(argus_quant_block_size(99999) == 0);

    // Test new logit bias sampling null checks
    assert(argus_sample_token_with_bias(nullptr, 0, 0.0f, 0.0f, nullptr, 0) == -1);

    // Verify context params struct alignment size and u_batch configuration
    std::cout << "[Test] Verifying struct alignment layout size and sequence parameters..." << std::endl;
    assert(sizeof(argus_context_params_t) == 40);

    argus_context_params_t test_params = {};
    test_params.context_length = 2048;
    test_params.u_batch = 1024;
    test_params.n_seq_max = 2;
    test_params.embeddings = true;
    test_params.kv_unified = true;
    assert(test_params.u_batch == 1024);
    assert(test_params.n_seq_max == 2);
    assert(test_params.embeddings == true);
    assert(test_params.kv_unified == true);

    std::cout << "[Test] Vocab and metadata null safety tests completed successfully." << std::endl;

    // Test dynamic thread count control null checks
    std::cout << "[Test] Verifying thread count control null safety checks..." << std::endl;
    argus_set_n_threads(nullptr, 4, 4);
    assert(argus_get_n_threads(nullptr) == -1);
    assert(argus_get_n_threads_batch(nullptr) == -1);
    argus_audio_set_n_threads(nullptr, 4);
    assert(argus_audio_get_n_threads(nullptr) == -1);
    std::cout << "[Test] Thread count control null safety assertions completed successfully." << std::endl;


    // 7. Free the global backends
    argus_backend_free();
    std::cout << "[Test] Backend freed successfully." << std::endl;

    std::cout << "[Test] All integrated assertions passed!" << std::endl;
    return 0;
}
