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
    assert(std::strcmp(argus_version(), "0.2.3") == 0);

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

    // Test new logit bias sampling null checks
    assert(argus_sample_token_with_bias(nullptr, 0, 0.0f, 0.0f, nullptr, nullptr, 0) == -1);

    std::cout << "[Test] Vocab and metadata null safety tests completed successfully." << std::endl;

    // 7. Free the global backends
    argus_backend_free();
    std::cout << "[Test] Backend freed successfully." << std::endl;

    std::cout << "[Test] All integrated assertions passed!" << std::endl;
    return 0;
}
