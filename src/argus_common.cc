/**
 * @file argus_common.cc
 * @brief Global lifecycle orchestration and hardware backend initialization core.
 *
 * Implements the process-global state control hooks declared in libargus.h,
 * establishing a thread-safe barrier ensuring coordinated device loading.
 */

#include "libargus.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <mutex>
#include <cstdio>

// Process-global synchronization structures
static std::mutex g_backend_mutex;
static bool       g_backend_initialized = false;
static int32_t    g_active_context_count = 0;

extern "C" {

bool argus_backend_init(const char * custom_plugin_path) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);

    if (g_backend_initialized) {
        g_active_context_count++;
        return true;
    }

    // Initialize the process-wide physical and virtual device backends
#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)
    // Dynamic loading pass for compiled runtime variants
    if (custom_plugin_path && custom_plugin_path[0] != '\0') {
        ggml_backend_load_all_from_path(custom_plugin_path);
    } else {
        ggml_backend_load_all();
    }
#endif

    // Bootstrap primary transformer execution runtime properties
    llama_backend_init();

    g_backend_initialized = true;
    g_active_context_count = 1;

    return true;
}

void argus_backend_free(void) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);

    if (!g_backend_initialized) {
        return;
    }

    g_active_context_count--;

    // Explicitly release unmanaged drivers only when all active instances clear
    if (g_active_context_count <= 0) {
        llama_backend_free();
        g_backend_initialized = false;
        g_active_context_count = 0;
    }
}

int32_t argus_backend_get_count(void) {
    return (int32_t)ggml_backend_dev_count();
}

const char * argus_backend_get_name(int32_t index) {
    if (index < 0 || index >= (int32_t)ggml_backend_dev_count()) {
        return nullptr;
    }
    ggml_backend_dev_t device = ggml_backend_dev_get((size_t)index);
    if (!device) {
        return nullptr;
    }
    return ggml_backend_dev_name(device);
}

const char * argus_version(void) {
    // Returns the compile-time injected version string
    return LIBARGUS_VERSION;
}

} // extern "C"