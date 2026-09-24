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
#include "whisper.h"
#include "mtmd.h"
#include "argus_abi.h"
#include <cstddef>
#include <cstdlib>
#include <algorithm>

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

static_assert(sizeof(argus_model_params_t) == 16);
static_assert(sizeof(argus_context_params_t) == 40);
static_assert(sizeof(argus_token_batch_t) == 32);
static_assert(offsetof(argus_token_batch_t, abort_flag) == 24);
static_assert(sizeof(argus_multimodal_params_t) == 16);
static_assert(sizeof(argus_logit_bias_t) == 8);
static_assert(sizeof(argus_sampler_params_t) == 56);
static_assert(offsetof(argus_sampler_params_t, seed) == 48);
static_assert(sizeof(argus_perf_timings_t) == 48);
static_assert(alignof(argus_perf_timings_t) == 8);
static_assert(offsetof(argus_perf_timings_t, t_start_ms) == 0);
static_assert(offsetof(argus_perf_timings_t, t_load_ms) == 8);
static_assert(offsetof(argus_perf_timings_t, t_p_eval_ms) == 16);
static_assert(offsetof(argus_perf_timings_t, t_eval_ms) == 24);
static_assert(offsetof(argus_perf_timings_t, n_p_eval) == 32);
static_assert(offsetof(argus_perf_timings_t, n_eval) == 36);
static_assert(offsetof(argus_perf_timings_t, n_reused) == 40);
static_assert(offsetof(argus_perf_timings_t, reserved_padding) == 44);

// =========================================================================
// Global Diagnostic Logging State & Dispatch
// =========================================================================

static std::atomic<argus_log_level_t>    g_log_level{ARGUS_LOG_WARN};
static std::atomic<bool>                 g_log_level_explicit{false};
static std::atomic<argus_log_callback_t> g_log_callback{nullptr};
static std::atomic<void *>               g_log_user_data{nullptr};
static std::mutex                        g_log_mutex;
static bool                              g_stderr_needs_newline{false};

// Thread-local continuation tracking guarantees zero-race condition across concurrent worker threads
static thread_local argus_log_level_t    tl_last_emitted_level{ARGUS_LOG_NONE};

static void argus_internal_log_dispatch(enum ggml_log_level ggml_level, const char * text, void * /*user_data*/) {
    if (!text || text[0] == '\0') {
        return;
    }

    argus_log_level_t active_threshold = g_log_level.load(std::memory_order_relaxed);
    if (active_threshold == ARGUS_LOG_NONE) {
        return;
    }

    argus_log_level_t mapped_level;
    switch (ggml_level) {
        case GGML_LOG_LEVEL_DEBUG: mapped_level = ARGUS_LOG_DEBUG; break;
        case GGML_LOG_LEVEL_INFO:  mapped_level = ARGUS_LOG_INFO;  break;
        case GGML_LOG_LEVEL_WARN:  mapped_level = ARGUS_LOG_WARN;  break;
        case GGML_LOG_LEVEL_ERROR: mapped_level = ARGUS_LOG_ERROR; break;
        case GGML_LOG_LEVEL_CONT:  mapped_level = ARGUS_LOG_CONT;  break;
        default:                   mapped_level = ARGUS_LOG_NONE;  break;
    }

    // Continuation chunk handling: inherit the severity of the primary line for this thread
    if (mapped_level == ARGUS_LOG_CONT) {
        if (tl_last_emitted_level < active_threshold || tl_last_emitted_level == ARGUS_LOG_NONE) {
            return;
        }
    } else {
        tl_last_emitted_level = mapped_level;
        if (mapped_level < active_threshold) {
            return;
        }
    }

    // Message passed threshold: serialize emission to prevent torn formatting or multi-threaded callback races
    std::lock_guard<std::mutex> lock(g_log_mutex);
    auto cb = g_log_callback.load(std::memory_order_acquire);
    auto ud = g_log_user_data.load(std::memory_order_acquire);
    if (cb) {
        try {
            cb(mapped_level, text, ud);
        } catch (...) {
            // Foreign exception barrier
        }
    } else {
        if (mapped_level != ARGUS_LOG_CONT && g_stderr_needs_newline) {
            std::fputc('\n', stderr);
            g_stderr_needs_newline = false;
        }
        std::fputs(text, stderr);
        size_t len = std::strlen(text);
        if (len > 0) {
            if ((mapped_level == ARGUS_LOG_WARN || mapped_level == ARGUS_LOG_ERROR) && text[len - 1] != '\n') {
                std::fputc('\n', stderr);
                g_stderr_needs_newline = false;
            } else {
                g_stderr_needs_newline = (text[len - 1] != '\n');
            }
        }
        std::fflush(stderr);
    }
}

static void argus_init_log_level_from_env() {
    if (g_log_level_explicit.load(std::memory_order_acquire)) {
        return; // Programmatic configuration takes precedence
    }

    const char * env = nullptr;
#if defined(__linux__) && defined(_GNU_SOURCE)
    env = secure_getenv("LIBARGUS_LOG_LEVEL");
    if (!env) env = secure_getenv("ARGUS_LOG_LEVEL");
#else
    env = std::getenv("LIBARGUS_LOG_LEVEL");
    if (!env) env = std::getenv("ARGUS_LOG_LEVEL");
#endif

    if (!env || env[0] == '\0') {
        g_log_level.store(ARGUS_LOG_WARN, std::memory_order_relaxed);
        return;
    }

    // Case-insensitive ASCII comparison helper
    auto iequals = [](const char * a, const char * b) {
        while (*a && *b) {
            char ca = (*a >= 'A' && *a <= 'Z') ? (char)(*a + 32) : *a;
            char cb = (*b >= 'A' && *b <= 'Z') ? (char)(*b + 32) : *b;
            if (ca != cb) return false;
            ++a; ++b;
        }
        return *a == *b;
    };

    if (iequals(env, "NONE") || std::strcmp(env, "0") == 0 || iequals(env, "OFF") || iequals(env, "SILENT")) {
        g_log_level.store(ARGUS_LOG_NONE, std::memory_order_relaxed);
    } else if (iequals(env, "DEBUG") || std::strcmp(env, "1") == 0 || iequals(env, "TRACE")) {
        g_log_level.store(ARGUS_LOG_DEBUG, std::memory_order_relaxed);
    } else if (iequals(env, "INFO") || std::strcmp(env, "2") == 0) {
        g_log_level.store(ARGUS_LOG_INFO, std::memory_order_relaxed);
    } else if (iequals(env, "WARN") || iequals(env, "WARNING") || std::strcmp(env, "3") == 0) {
        g_log_level.store(ARGUS_LOG_WARN, std::memory_order_relaxed);
    } else if (iequals(env, "ERROR") || std::strcmp(env, "4") == 0 || iequals(env, "ERR")) {
        g_log_level.store(ARGUS_LOG_ERROR, std::memory_order_relaxed);
    }
}

static void argus_register_upstream_log_hooks() {
    static std::once_flag s_once;
    std::call_once(s_once, []() {
        argus_init_log_level_from_env();
        llama_log_set(argus_internal_log_dispatch, nullptr);
        ggml_log_set(argus_internal_log_dispatch, nullptr);
        whisper_log_set(argus_internal_log_dispatch, nullptr);
        mtmd_log_set(argus_internal_log_dispatch, nullptr);
    });
}

namespace {
struct ArgusLogInitTrigger {
    ArgusLogInitTrigger() {
        argus_register_upstream_log_hooks();
    }
};
static ArgusLogInitTrigger s_log_init_trigger;
}

extern "C" {

int32_t argus_build_info_copy(char * out, int32_t capacity) {
    if (capacity < 0 || (!out && capacity != 0)) return -1;
    const auto & info = argus_diagnostic_info();
    const int32_t length = info.length;
    if (length < 0) return -1;
    if (capacity > 0) {
        const int32_t count = std::min(length, capacity - 1);
        std::memcpy(out, info.data.data(), count);
        out[count] = '\0';
    }
    return length;
}


argus_error_code_t argus_last_error_code(void) {
    return tl_last_error.code;
}

const char * argus_last_error_message(void) {
    return tl_last_error.message;
}

int32_t argus_last_error_message_copy(char * out, int32_t capacity) {
    if (!out || capacity <= 0) {
        return -1;
    }
    size_t len = std::strlen(tl_last_error.message);
    size_t to_copy = (len < (size_t)(capacity - 1)) ? len : (size_t)(capacity - 1);
    std::memcpy(out, tl_last_error.message, to_copy);
    out[to_copy] = '\0';
    return (int32_t)len;
}

void argus_clear_error(void) {
    clear_last_error();
}

uint64_t argus_build_features(void) {
    uint64_t features = ARGUS_FEATURE_CPU;
#if defined(ARGUS_BUILD_CUDA)
    features |= ARGUS_FEATURE_CUDA;
#endif
#if defined(ARGUS_BUILD_HIP)
    features |= ARGUS_FEATURE_HIP;
#endif
#if defined(ARGUS_BUILD_VULKAN)
    features |= ARGUS_FEATURE_VULKAN;
#endif
#if defined(ARGUS_BUILD_METAL)
    features |= ARGUS_FEATURE_METAL;
#endif
    return features;
}

bool argus_backend_init(const char * custom_plugin_path) {
    try {
        clear_last_error();
        argus_register_upstream_log_hooks();
        std::lock_guard<std::mutex> lock(g_backend_mutex);

        if (g_backend_initialized) {
            g_active_context_count++;
            g_backend_teardown_pending = false;
            return true;
        }

        // Compiled backends register when llama_backend_init() queries the
        // process registry. Only scan the caller's explicit directory; default
        // filesystem discovery is unnecessary for this statically linked build
        // and would make initialization depend on ambient executable contents.
        if (custom_plugin_path && custom_plugin_path[0] != '\0') {
            ggml_backend_load_all_from_path(custom_plugin_path);
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

        {
            std::lock_guard<std::mutex> log_lock(g_log_mutex);
            g_stderr_needs_newline = false;
        }
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown native exception during backend free");
    }
}

bool argus_backend_is_initialized(void) {
    try {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        return g_backend_initialized;
    } catch (const std::exception & e) {
        set_last_error(ARGUS_ERROR_INTERNAL, e.what());
    } catch (...) {
        set_last_error(ARGUS_ERROR_INTERNAL, "unknown exception while querying backend state");
    }
    return false;
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

ARGUS_API void argus_set_log_level(argus_log_level_t level) {
    argus_register_upstream_log_hooks();
    if (level == ARGUS_LOG_NONE) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_stderr_needs_newline = false;
    }
    g_log_level_explicit.store(true, std::memory_order_release);
    g_log_level.store(level, std::memory_order_release);
}

ARGUS_API argus_log_level_t argus_get_log_level(void) {
    argus_register_upstream_log_hooks();
    return g_log_level.load(std::memory_order_acquire);
}

ARGUS_API void argus_set_log_callback(argus_log_callback_t callback, void * user_data) {
    argus_register_upstream_log_hooks();
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_stderr_needs_newline = false;
    g_log_user_data.store(user_data, std::memory_order_release);
    g_log_callback.store(callback, std::memory_order_release);
}

} // extern "C"
