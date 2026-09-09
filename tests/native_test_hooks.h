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
}
void argus_test_allocation_checkpoint();
