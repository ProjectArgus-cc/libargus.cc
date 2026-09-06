/**
 * @file argus_common.cc
 * @brief Global lifecycle orchestration and hardware backend initialization core.
 *
 * Implements the process-global state control hooks declared in libargus.h,
 * establishing a thread-safe barrier ensuring coordinated device loading,
 * structured error tracking, and deferred driver teardown.
 */

#include "libargus.h"
#include "argus_internal.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

// =========================================================================
// Thread-Local Zero-Allocation Diagnostic State
// =========================================================================

struct argus_error_state {
    argus_error_code_t code = ARGUS_SUCCESS;
    char message[512] = {0};
};

static thread_local argus_error_state tl_last_error;

void set_last_error(argus_error_code_t code, const char * msg) {
    tl_last_error.code = code;
    if (msg) {
        std::strncpy(tl_last_error.message, msg, sizeof(tl_last_error.message) - 1);
        tl_last_error.message[sizeof(tl_last_error.message) - 1] = '\0';
    } else {
        tl_last_error.message[0] = '\0';
    }
}

void clear_last_error() {
    tl_last_error.code = ARGUS_SUCCESS;
    tl_last_error.message[0] = '\0';
}

// =========================================================================
// Backend Lifecycle & Deferred Teardown State
// =========================================================================

static std::mutex            g_backend_mutex;
static bool                  g_backend_initialized = false;
static int32_t               g_active_context_count = 0;
static std::atomic<uint32_t> g_active_native_resources{0};
static bool                  g_backend_teardown_pending = false;

void argus_backend_resource_inc() {
    g_active_native_resources.fetch_add(1, std::memory_order_relaxed);
}

void argus_backend_resource_dec() {
    uint32_t prev = g_active_native_resources.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        if (g_backend_teardown_pending && g_active_context_count <= 0 && g_backend_initialized) {
            llama_backend_free();
            g_backend_initialized = false;
            g_backend_teardown_pending = false;
        }
    }
}

extern "C" {

argus_error_code_t argus_last_error_code(void) {
    return tl_last_error.code;
}

const char * argus_last_error_message(void) {
    return tl_last_error.message;
}

void argus_clear_error(void) {
    clear_last_error();
}

uint64_t argus_build_features(void) {
    uint64_t features = ARGUS_FEATURE_CPU;
#if defined(GGML_USE_CUDA)
    features |= ARGUS_FEATURE_CUDA;
#endif
#if defined(GGML_USE_HIP)
    features |= ARGUS_FEATURE_HIP;
#endif
#if defined(GGML_USE_VULKAN)
    features |= ARGUS_FEATURE_VULKAN;
#endif
#if defined(GGML_USE_METAL)
    features |= ARGUS_FEATURE_METAL;
#endif
    return features;
}

bool argus_backend_init(const char * custom_plugin_path) {
    try {
        clear_last_error();
        std::lock_guard<std::mutex> lock(g_backend_mutex);

        if (g_backend_initialized) {
            g_active_context_count++;
            g_backend_teardown_pending = false;
            return true;
        }

        // Initialize the process-wide physical and virtual device backends
        if (custom_plugin_path && custom_plugin_path[0] != '\0') {
            ggml_backend_load_all_from_path(custom_plugin_path);
        } else {
            ggml_backend_load_all();
        }

        // Bootstrap primary transformer execution runtime properties
        llama_backend_init();

        g_backend_initialized = true;
        g_backend_teardown_pending = false;
        g_active_context_count = 1;

        return true;
    } catch (const std::bad_alloc & e) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_BACKEND, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during backend initialization");
    }
    return false;
}

void argus_backend_free(void) {
    try {
        clear_last_error();
        std::lock_guard<std::mutex> lock(g_backend_mutex);

        if (!g_backend_initialized) {
            return;
        }

        g_active_context_count--;

        // Only tear down drivers when explicit init leases clear
        if (g_active_context_count <= 0) {
            g_active_context_count = 0;
            // If live models or contexts still exist, defer physical driver teardown
            if (g_active_native_resources.load(std::memory_order_acquire) > 0) {
                g_backend_teardown_pending = true;
            } else {
                llama_backend_free();
                g_backend_initialized = false;
                g_backend_teardown_pending = false;
            }
        }
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during backend free");
    }
}

bool argus_backend_is_initialized(void) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    return g_backend_initialized;
}

int32_t argus_backend_get_count(void) {
    try {
        clear_last_error();
        return (int32_t)ggml_backend_dev_count();
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "failed to query backend device count");
        return 0;
    }
}

const char * argus_backend_get_name(int32_t index) {
    try {
        clear_last_error();
        if (index < 0 || index >= (int32_t)ggml_backend_dev_count()) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "backend index out of range");
            return nullptr;
        }
        ggml_backend_dev_t device = ggml_backend_dev_get((size_t)index);
        if (!device) {
            return nullptr;
        }
        return ggml_backend_dev_name(device);
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "failed to query backend device name");
        return nullptr;
    }
}

const char * argus_version(void) {
    return LIBARGUS_VERSION;
}

} // extern "C"