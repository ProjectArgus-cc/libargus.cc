#pragma once
#include "libargus.h"
// Only linked into argus_test. Callers unregister observers after joining workers.
extern "C" {
ARGUS_API void argus_test_set_observer(void (*observer)(int, const void *));
ARGUS_API argus_multimodal_t * argus_test_projector(argus_model_t * model);
ARGUS_API argus_input_chunks_t * argus_test_chunks(void);
ARGUS_API argus_video_t * argus_test_video(argus_multimodal_t * mctx, int count);
ARGUS_API void argus_test_set_eval_result(int result);
ARGUS_API void argus_test_fail_after_retain(bool fail);
ARGUS_API void argus_test_emit_log(int ggml_level, const char * text);
ARGUS_API void argus_test_set_force_rollback_fail(bool fail);
ARGUS_API int32_t argus_test_sampler_get_history_token(const argus_context_t * ctx, int32_t seq_id, int32_t idx);
ARGUS_API int32_t argus_test_sampler_get_history_kv_pos(const argus_context_t * ctx, int32_t seq_id, int32_t idx);
ARGUS_API int32_t argus_test_draft_seq_pos_max(const argus_context_t * ctx, int32_t seq_id);
ARGUS_API bool argus_test_has_logits(const argus_context_t * ctx, int32_t seq_id);
ARGUS_API bool argus_test_has_embeddings(const argus_context_t * ctx, int32_t seq_id);
ARGUS_API void argus_test_set_batch_alloc_fail_step(int32_t step);
ARGUS_API void argus_test_set_context_batch_scratch_enabled(argus_context_t * ctx, bool enabled);
ARGUS_API argus_audio_context_t * argus_test_audio(const char * transcript);
ARGUS_API void argus_test_set_vocoder_override(bool allow);
}
void argus_test_allocation_checkpoint();

