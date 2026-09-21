#include "native_test_hooks.h"
#include <atomic>
#include <new>
static std::atomic<bool> fail_after_retain{false};
extern "C" void argus_test_fail_after_retain(bool fail) { fail_after_retain = fail; }
void argus_test_allocation_checkpoint() {
    if (fail_after_retain.load()) throw std::bad_alloc();
}
static std::atomic<void (*)(int, const void *)> observer{nullptr};
extern "C" void argus_test_set_observer(void (*value)(int, const void *)) {
    observer.store(value, std::memory_order_release);
}
void argus_test_notify(int event, const void * resource) {
    if (auto callback = observer.load(std::memory_order_acquire)) callback(event, resource);
}

#include "ggml.h"
extern "C" void argus_test_emit_log(int ggml_level, const char * text) {
    ggml_log_callback cb = nullptr;
    void * user_data = nullptr;
    ggml_log_get(&cb, &user_data);
    if (cb) {
        cb((enum ggml_log_level)ggml_level, text, user_data);
    }
}
