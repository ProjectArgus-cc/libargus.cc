/**
 * @file argus_jni.cc
 * @brief Zero-pinning JNI bridge compiled directly into libargus.
 *
 * Implements the native backing methods for cc.projectargus.libargus.internal.ArgusJni
 * targeting JDK 17+, JDK 21 LTS, and Android ART (API 26+) while delegating all execution
 * to the canonical libargus C ABI.
 *
 * Key architectural guarantees:
 * 1. Single-binary linkage: Compiled directly into libargus so JNI and C ABI share the exact
 *    same process-global backend state and thread-local error storage (tl_last_error).
 * 2. Zero GetPrimitiveArrayCritical: Never pins the Java/Android GC heap across native compute.
 *    Uses bounded stack buffers + Get<Type>ArrayRegion / Set<Type>ArrayRegion or DirectByteBuffer.
 * 3. Standard UTF-8 byte[] transport: Avoids JNI Modified UTF-8 (CESU-8 6-byte surrogates) and
 *    Android ART CheckJNI traps on 4-byte code points or partial BPE token byte slices.
 * 4. Two-tier adaptive buffers: Stack-first fast path with exact-size dynamic vector retry for
 *    both negative-return (argus_token_to_piece) and snprintf-style positive-return metadata APIs.
 * 5. Overflow-safe 64-bit spatial boundary validation on all array and DirectByteBuffer slices.
 */

#include <jni.h>
#include "libargus.h"
#include "argus_internal.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define ARGUS_JNI_EXPORT JNIEXPORT
#elif defined(__GNUC__) || defined(__clang__)
  #define ARGUS_JNI_EXPORT __attribute__((visibility("default"))) JNIEXPORT
#else
  #define ARGUS_JNI_EXPORT JNIEXPORT
#endif

namespace {

static std::atomic<JavaVM *> g_jvm{nullptr};
static std::mutex            g_jni_log_mutex;
static jclass                g_jni_class = nullptr;
static jmethodID             g_on_log_mid = nullptr;

inline jbyteArray bytes_from_utf8(JNIEnv * env, const char * data, size_t len) noexcept {
    if (!env || !data || len > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        return nullptr;
    }
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));
    if (!arr) {
        return nullptr;
    }
    if (len > 0) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len), reinterpret_cast<const jbyte *>(data));
    }
    return arr;
}

inline jbyteArray cstr_to_jbytes(JNIEnv * env, const char * str) noexcept {
    if (!str) {
        return nullptr;
    }
    return bytes_from_utf8(env, str, std::strlen(str));
}

struct utf8_span {
    char             stack_buf[512];
    std::vector<char> heap_buf;
    const char *     ptr = nullptr;
    size_t           len = 0;
    bool             valid = false;

    utf8_span(JNIEnv * env, jbyteArray bytes, bool require_null_term = true) noexcept {
        if (!env || !bytes) {
            return;
        }
        const jsize jlen = env->GetArrayLength(bytes);
        if (jlen < 0) {
            return;
        }
        len = static_cast<size_t>(jlen);
        const size_t needed = require_null_term ? (len + 1) : (len > 0 ? len : 1);
        char * dst = nullptr;
        if (needed <= sizeof(stack_buf)) {
            dst = stack_buf;
        } else {
            try {
                heap_buf.resize(needed);
                dst = heap_buf.data();
            } catch (...) {
                set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate JNI UTF-8 buffer");
                return;
            }
        }
        if (jlen > 0) {
            env->GetByteArrayRegion(bytes, 0, jlen, reinterpret_cast<jbyte *>(dst));
            if (env->ExceptionCheck()) {
                return;
            }
        }
        if (require_null_term) {
            dst[len] = '\0';
        }
        ptr = dst;
        valid = true;
    }
};

// Helper for snprintf-style C API functions that return total required length (>= buf_size on truncation)
// and negative on failure.
template <typename Fn>
jbyteArray call_snprintf_string_api(JNIEnv * env, Fn && fn) noexcept {
    char stack_buf[512];
    int32_t len = fn(stack_buf, static_cast<int32_t>(sizeof(stack_buf)));
    if (len < 0) {
        return nullptr;
    }
    if (len < static_cast<int32_t>(sizeof(stack_buf))) {
        return bytes_from_utf8(env, stack_buf, static_cast<size_t>(len));
    }
    if (len == std::numeric_limits<int32_t>::max()) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "string length exceeds int32 capacity");
        return nullptr;
    }
    try {
        std::vector<char> dyn_buf(static_cast<size_t>(len) + 1);
        int32_t retry_len = fn(dyn_buf.data(), static_cast<int32_t>(dyn_buf.size()));
        if (retry_len < 0) {
            return nullptr;
        }
        const size_t actual_len = static_cast<size_t>(retry_len) < dyn_buf.size()
            ? static_cast<size_t>(retry_len)
            : (dyn_buf.size() - 1);
        return bytes_from_utf8(env, dyn_buf.data(), actual_len);
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate dynamic string buffer");
        return nullptr;
    }
}

bool copy_logit_biases(
    JNIEnv * env,
    jintArray biasTokens,
    jfloatArray biasValues,
    jint biasCount,
    std::vector<argus_logit_bias_t> & out_biases
) noexcept {
    if (biasCount < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "negative biasCount");
        return false;
    }
    if (biasCount == 0) {
        return true;
    }
    if (!biasTokens || !biasValues) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null bias array with positive biasCount");
        return false;
    }
    if (env->GetArrayLength(biasTokens) < biasCount || env->GetArrayLength(biasValues) < biasCount) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "bias array length smaller than biasCount");
        return false;
    }
    try {
        std::vector<jint> tokens(static_cast<size_t>(biasCount));
        std::vector<jfloat> values(static_cast<size_t>(biasCount));
        env->GetIntArrayRegion(biasTokens, 0, biasCount, tokens.data());
        env->GetFloatArrayRegion(biasValues, 0, biasCount, values.data());
        if (env->ExceptionCheck()) {
            return false;
        }
        out_biases.resize(static_cast<size_t>(biasCount));
        for (jint i = 0; i < biasCount; ++i) {
            out_biases[static_cast<size_t>(i)].token = static_cast<int32_t>(tokens[static_cast<size_t>(i)]);
            out_biases[static_cast<size_t>(i)].bias  = static_cast<float>(values[static_cast<size_t>(i)]);
        }
        return true;
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate logit bias buffer");
        return false;
    }
}

void jni_log_trampoline(argus_log_level_t level, const char * text, void * /*user_data*/) {
    if (!text) {
        return;
    }
    JavaVM * jvm = g_jvm.load(std::memory_order_acquire);
    if (!jvm) {
        return;
    }
    jclass cls = nullptr;
    jmethodID mid = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_jni_log_mutex);
        cls = g_jni_class;
        mid = g_on_log_mid;
    }
    if (!cls || !mid) {
        return;
    }

    JNIEnv * env = nullptr;
    bool attached_here = false;
    jint status = jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
#if defined(__ANDROID__)
        if (jvm->AttachCurrentThreadAsDaemon(&env, nullptr) != JNI_OK || !env) {
            return;
        }
#else
        if (jvm->AttachCurrentThreadAsDaemon(reinterpret_cast<void **>(&env), nullptr) != JNI_OK || !env) {
            return;
        }
#endif
        attached_here = true;
    } else if (status != JNI_OK || !env) {
        return;
    }

    if (env->PushLocalFrame(8) == 0) {
        jbyteArray utf8_bytes = bytes_from_utf8(env, text, std::strlen(text));
        if (utf8_bytes && !env->ExceptionCheck()) {
            env->CallStaticVoidMethod(cls, mid, static_cast<jint>(level), utf8_bytes);
        }
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        env->PopLocalFrame(nullptr);
    }

    if (attached_here) {
        jvm->DetachCurrentThread();
    }
}

} // namespace

extern "C" {

ARGUS_JNI_EXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void * /*reserved*/) {
    g_jvm.store(vm, std::memory_order_release);
    return JNI_VERSION_1_6;
}

ARGUS_JNI_EXPORT void JNICALL JNI_OnUnload(JavaVM * vm, void * /*reserved*/) {
    argus_set_log_callback(nullptr, nullptr);
    JNIEnv * env = nullptr;
    if (vm && vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK && env) {
        std::lock_guard<std::mutex> lock(g_jni_log_mutex);
        if (g_jni_class) {
            env->DeleteGlobalRef(g_jni_class);
            g_jni_class = nullptr;
            g_on_log_mid = nullptr;
        }
    }
    g_jvm.store(nullptr, std::memory_order_release);
}

// =========================================================================
// 1. Diagnostics, Build Identity, Logging & Backend Lifecycle
// =========================================================================

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeLastErrorCode(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return static_cast<jint>(argus_last_error_code());
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeLastErrorMessage(
    JNIEnv * env, jclass /*cls*/) {
    const char * msg = argus_last_error_message();
    return cstr_to_jbytes(env, msg ? msg : "");
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeClearError(
    JNIEnv * /*env*/, jclass /*cls*/) {
    argus_clear_error();
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBuildFeatures(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return static_cast<jlong>(argus_build_features());
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBuildInfo(
    JNIEnv * env, jclass /*cls*/) {
    int32_t needed = argus_build_info_copy(nullptr, 0);
    if (needed < 0) {
        return nullptr;
    }
    return call_snprintf_string_api(env, [](char * buf, int32_t cap) {
        return argus_build_info_copy(buf, cap);
    });
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBackendInit(
    JNIEnv * env, jclass /*cls*/, jbyteArray customPluginPathUtf8) {
    if (!customPluginPathUtf8) {
        return argus_backend_init(nullptr) ? JNI_TRUE : JNI_FALSE;
    }
    utf8_span path(env, customPluginPathUtf8, true);
    if (!path.valid) {
        return JNI_FALSE;
    }
    return argus_backend_init(path.ptr) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBackendFree(
    JNIEnv * /*env*/, jclass /*cls*/) {
    argus_backend_free();
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBackendIsInitialized(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return argus_backend_is_initialized() ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBackendGetCount(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return static_cast<jint>(argus_backend_get_count());
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBackendGetName(
    JNIEnv * env, jclass /*cls*/, jint index) {
    const char * name = argus_backend_get_name(static_cast<int32_t>(index));
    return cstr_to_jbytes(env, name);
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVersion(
    JNIEnv * env, jclass /*cls*/) {
    return cstr_to_jbytes(env, argus_version());
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSetLogLevel(
    JNIEnv * /*env*/, jclass /*cls*/, jint level) {
    argus_set_log_level(static_cast<argus_log_level_t>(level));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeGetLogLevel(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return static_cast<jint>(argus_get_log_level());
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSetLogCallbackEnabled(
    JNIEnv * env, jclass cls, jboolean enabled) {
    if (!env) {
        return;
    }
    if (!g_jvm.load(std::memory_order_acquire)) {
        JavaVM * vm = nullptr;
        if (env->GetJavaVM(&vm) == JNI_OK && vm) {
            g_jvm.store(vm, std::memory_order_release);
        }
    }
    if (enabled == JNI_TRUE) {
        {
            std::lock_guard<std::mutex> lock(g_jni_log_mutex);
            if (!g_jni_class && cls) {
                g_jni_class = reinterpret_cast<jclass>(env->NewGlobalRef(cls));
            }
            if (g_jni_class && !g_on_log_mid) {
                g_on_log_mid = env->GetStaticMethodID(g_jni_class, "onNativeLog", "(I[B)V");
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    g_on_log_mid = nullptr;
                }
            }
        }
        argus_set_log_callback(jni_log_trampoline, nullptr);
    } else {
        argus_set_log_callback(nullptr, nullptr);
    }
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeLogInFlightCount(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return static_cast<jint>(argus_log_in_flight_count());
}

// =========================================================================
// 2. Model Lifecycle, Vocabulary & GGUF Introspection
// =========================================================================

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelLoad(
    JNIEnv * env, jclass /*cls*/, jbyteArray pathUtf8, jint gpuLayers, jboolean useMlock) {
    if (!pathUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "model path is null");
        return 0L;
    }
    utf8_span path(env, pathUtf8, true);
    if (!path.valid) {
        return 0L;
    }
    argus_model_params_t params = {};
    params.model_path = path.ptr;
    params.gpu_layers = static_cast<int32_t>(gpuLayers);
    params.use_mlock  = (useMlock == JNI_TRUE);
    return reinterpret_cast<jlong>(argus_model_load(&params));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelLoadFd(
    JNIEnv * /*env*/, jclass /*cls*/, jint fd, jint gpuLayers, jboolean useMlock) {
    argus_model_params_t params = {};
    params.model_path = nullptr;
    params.gpu_layers = static_cast<int32_t>(gpuLayers);
    params.use_mlock  = (useMlock == JNI_TRUE);
    return reinterpret_cast<jlong>(argus_model_load_fd(static_cast<int32_t>(fd), &params));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelRetain(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_retain(reinterpret_cast<argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelRelease(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    argus_model_release(reinterpret_cast<argus_model_t *>(modelHandle));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    argus_model_free(reinterpret_cast<argus_model_t *>(modelHandle));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeTokenize(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jbyteArray utf8Text, jintArray outTokens, jint maxTokens, jboolean addBos) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model || !utf8Text || maxTokens < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid tokenize arguments");
        return -1;
    }
    if (maxTokens > 0) {
        if (!outTokens || env->GetArrayLength(outTokens) < maxTokens) {
            set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "outTokens array is null or smaller than maxTokens");
            return -1;
        }
    }
    utf8_span text(env, utf8Text, true);
    if (!text.valid) {
        return -1;
    }
    if (maxTokens == 0) {
        return static_cast<jint>(argus_tokenize_n(model, text.ptr, text.len, nullptr, 0, addBos == JNI_TRUE));
    }

    int32_t stack_tokens[512];
    std::vector<int32_t> heap_tokens;
    int32_t * tok_ptr = stack_tokens;
    if (static_cast<size_t>(maxTokens) > (sizeof(stack_tokens) / sizeof(stack_tokens[0]))) {
        try {
            heap_tokens.resize(static_cast<size_t>(maxTokens));
            tok_ptr = heap_tokens.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate tokenization output buffer");
            return -1;
        }
    }

    int32_t written = argus_tokenize_n(model, text.ptr, text.len, tok_ptr, static_cast<int32_t>(maxTokens), addBos == JNI_TRUE);
    if (written > 0) {
        const jsize to_copy = static_cast<jsize>(written <= maxTokens ? written : maxTokens);
        env->SetIntArrayRegion(outTokens, 0, to_copy, reinterpret_cast<const jint *>(tok_ptr));
    }
    return static_cast<jint>(written);
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeTokenToPiece(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jint token) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model handle");
        return nullptr;
    }
    char stack_buf[256];
    int32_t written = argus_token_to_piece(model, static_cast<int32_t>(token), stack_buf, static_cast<int32_t>(sizeof(stack_buf)));
    if (written >= 0) {
        return bytes_from_utf8(env, stack_buf, static_cast<size_t>(written));
    }
    // Check if -1 was due to an invalid argument / out-of-bounds token vs buffer overflow
    if (argus_last_error_code() != ARGUS_SUCCESS) {
        return nullptr;
    }
    const int64_t needed = -static_cast<int64_t>(written);
    if (needed <= 0 || needed >= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        set_last_error(ARGUS_ERROR_INTERNAL, "invalid piece length returned by tokenizer");
        return nullptr;
    }
    try {
        std::vector<char> dyn_buf(static_cast<size_t>(needed) + 1);
        written = argus_token_to_piece(model, static_cast<int32_t>(token), dyn_buf.data(), static_cast<int32_t>(dyn_buf.size()));
        if (written < 0) {
            return nullptr;
        }
        return bytes_from_utf8(env, dyn_buf.data(), static_cast<size_t>(written));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate dynamic token piece buffer");
        return nullptr;
    }
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeTokenToPieceInto(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jint token, jbyteArray outBuf, jint offset, jint length) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model || !outBuf || offset < 0 || length <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid tokenToPieceInto arguments");
        return -1;
    }
    const jsize arr_len = env->GetArrayLength(outBuf);
    if (static_cast<int64_t>(offset) + static_cast<int64_t>(length) > static_cast<int64_t>(arr_len)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "outBuf slice [offset, offset+length) exceeds array length");
        return -1;
    }

    char stack_buf[512];
    std::vector<char> heap_buf;
    char * buf_ptr = stack_buf;
    if (static_cast<size_t>(length) > sizeof(stack_buf)) {
        try {
            heap_buf.resize(static_cast<size_t>(length));
            buf_ptr = heap_buf.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate token piece buffer");
            return -1;
        }
    }

    int32_t written = argus_token_to_piece(model, static_cast<int32_t>(token), buf_ptr, static_cast<int32_t>(length));
    if (written > 0) {
        env->SetByteArrayRegion(outBuf, offset, static_cast<jsize>(written), reinterpret_cast<const jbyte *>(buf_ptr));
    }
    return static_cast<jint>(written);
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabBos(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_vocab_bos(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabEos(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_vocab_eos(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabEot(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_vocab_eot(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabPad(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_vocab_pad(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabNTokens(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_vocab_n_tokens(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVocabIsEog(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle, jint token) {
    return argus_vocab_is_eog(reinterpret_cast<const argus_model_t *>(modelHandle), static_cast<int32_t>(token))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelMetaValStr(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jbyteArray keyUtf8) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model || !keyUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model or metadata key");
        return nullptr;
    }
    utf8_span key(env, keyUtf8, true);
    if (!key.valid) {
        return nullptr;
    }
    return call_snprintf_string_api(env, [&](char * buf, int32_t cap) {
        return argus_model_meta_val_str_n(model, key.ptr, key.len, buf, cap);
    });
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelMetaCount(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_meta_count(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelMetaKeyByIndex(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jint index) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model handle");
        return nullptr;
    }
    return call_snprintf_string_api(env, [&](char * buf, int32_t cap) {
        return argus_model_meta_key_by_index(model, static_cast<int32_t>(index), buf, cap);
    });
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelMetaValStrByIndex(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jint index) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model handle");
        return nullptr;
    }
    return call_snprintf_string_api(env, [&](char * buf, int32_t cap) {
        return argus_model_meta_val_str_by_index(model, static_cast<int32_t>(index), buf, cap);
    });
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNEmbd(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_embd(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNEmbdOut(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_embd_out(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNEmbdInp(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_embd_inp(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNCtxTrain(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_ctx_train(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNLayer(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_layer(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNHead(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_head(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNHeadKv(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_head_kv(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNParams(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jlong>(argus_model_n_params(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelHasEncoder(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_has_encoder(reinterpret_cast<const argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelNPosPerEmbd(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jint>(argus_model_n_pos_per_embd(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelIsMRoPE(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_is_mrope(reinterpret_cast<const argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelIsRecurrent(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_is_recurrent(reinterpret_cast<const argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelIsHybrid(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_is_hybrid(reinterpret_cast<const argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelIsDiffusion(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return argus_model_is_diffusion(reinterpret_cast<const argus_model_t *>(modelHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelSize(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle) {
    return static_cast<jlong>(argus_model_size(reinterpret_cast<const argus_model_t *>(modelHandle)));
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelDesc(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model handle");
        return nullptr;
    }
    return call_snprintf_string_api(env, [&](char * buf, int32_t cap) {
        return argus_model_desc(model, buf, cap);
    });
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelKvBytesPerToken(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle, jint typeK, jint typeV) {
    return static_cast<jlong>(argus_model_kv_bytes_per_token(
        reinterpret_cast<const argus_model_t *>(modelHandle),
        static_cast<int32_t>(typeK),
        static_cast<int32_t>(typeV)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeModelEstimateVramBytes(
    JNIEnv * /*env*/, jclass /*cls*/, jlong modelHandle, jint contextLength, jint typeK, jint typeV) {
    return static_cast<jlong>(argus_model_estimate_vram_bytes(
        reinterpret_cast<const argus_model_t *>(modelHandle),
        static_cast<int32_t>(contextLength),
        static_cast<int32_t>(typeK),
        static_cast<int32_t>(typeV)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeQuantTypeSize(
    JNIEnv * /*env*/, jclass /*cls*/, jint type) {
    return static_cast<jlong>(argus_quant_type_size(static_cast<int32_t>(type)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeQuantBlockSize(
    JNIEnv * /*env*/, jclass /*cls*/, jint type) {
    return static_cast<jint>(argus_quant_block_size(static_cast<int32_t>(type)));
}

// =========================================================================
// 3. Context Lifecycle, Telemetry, Decoding, Embeddings, Sampling & KV
// =========================================================================

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextInit(
    JNIEnv * /*env*/, jclass /*cls*/,
    jlong modelHandle,
    jlong draftModelHandle,
    jint contextLength,
    jint cpuThreads,
    jint typeK,
    jint typeV,
    jint specDraftNMax,
    jint uBatch,
    jint nSeqMax,
    jboolean enableDraftMtp,
    jboolean embeddings,
    jboolean kvUnified) {
    auto * model = reinterpret_cast<argus_model_t *>(modelHandle);
    if (!model) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model handle");
        return 0L;
    }
    argus_context_params_t params = {};
    params.draft_model      = reinterpret_cast<const argus_model_t *>(draftModelHandle);
    params.context_length   = static_cast<int32_t>(contextLength);
    params.cpu_threads      = static_cast<int32_t>(cpuThreads);
    params.type_k           = static_cast<int32_t>(typeK);
    params.type_v           = static_cast<int32_t>(typeV);
    params.spec_draft_n_max = static_cast<int32_t>(specDraftNMax);
    params.u_batch          = static_cast<int32_t>(uBatch);
    params.n_seq_max        = static_cast<int32_t>(nSeqMax);
    params.enable_draft_mtp = (enableDraftMtp == JNI_TRUE);
    params.embeddings       = (embeddings == JNI_TRUE);
    params.kv_unified       = (kvUnified == JNI_TRUE);
    return reinterpret_cast<jlong>(argus_context_init(model, &params));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    argus_context_free(reinterpret_cast<argus_context_t *>(ctxHandle));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextGetModel(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return reinterpret_cast<jlong>(argus_context_get_model(reinterpret_cast<const argus_context_t *>(ctxHandle)));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSetNThreads(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint nThreads, jint nThreadsBatch) {
    argus_set_n_threads(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(nThreads),
        static_cast<int32_t>(nThreadsBatch));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeGetNThreads(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return static_cast<jint>(argus_get_n_threads(reinterpret_cast<argus_context_t *>(ctxHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeGetNThreadsBatch(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return static_cast<jint>(argus_get_n_threads_batch(reinterpret_cast<argus_context_t *>(ctxHandle)));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextGetPerf(
    JNIEnv * env, jclass /*cls*/, jlong ctxHandle, jdoubleArray outDoubles, jintArray outInts) {
    const auto * ctx = reinterpret_cast<const argus_context_t *>(ctxHandle);
    if (!ctx || !outDoubles || !outInts ||
        env->GetArrayLength(outDoubles) < 4 || env->GetArrayLength(outInts) < 5) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid context or perf output arrays");
        return JNI_FALSE;
    }
    argus_perf_timings_t timings = {};
    if (!argus_context_get_perf(ctx, &timings)) {
        return JNI_FALSE;
    }
    const jdouble doubles[4] = {
        timings.t_start_ms,
        timings.t_load_ms,
        timings.t_p_eval_ms,
        timings.t_eval_ms
    };
    const jint ints[5] = {
        timings.n_p_eval,
        timings.n_eval,
        timings.n_reused,
        timings.n_reallocs_same,
        timings.n_reallocs_diff
    };
    env->SetDoubleArrayRegion(outDoubles, 0, 4, doubles);
    env->SetIntArrayRegion(outInts, 0, 5, ints);
    return JNI_TRUE;
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextResetPerf(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    argus_context_reset_perf(reinterpret_cast<argus_context_t *>(ctxHandle));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextSetCausalAttn(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jboolean causal) {
    return argus_context_set_causal_attn(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        causal == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextGetCausalAttn(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    bool out_causal = false;
    if (!argus_context_get_causal_attn(reinterpret_cast<const argus_context_t *>(ctxHandle), &out_causal)) {
        return -1;
    }
    return out_causal ? 1 : 0;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextHasDraft(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return argus_context_has_draft(reinterpret_cast<const argus_context_t *>(ctxHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeContextGetSeqMax(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return static_cast<jint>(argus_context_get_seq_max(reinterpret_cast<const argus_context_t *>(ctxHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeDecodeTokensArray(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jintArray tokens,
    jint offset,
    jint nTokens,
    jint startPos,
    jint seqId,
    jboolean requestLogits,
    jlong abortFlagHandle) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !tokens || offset < 0 || nTokens <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid decodeTokensArray parameters");
        return ARGUS_DECODE_ERROR;
    }
    const jsize arr_len = env->GetArrayLength(tokens);
    if (static_cast<int64_t>(offset) + static_cast<int64_t>(nTokens) > static_cast<int64_t>(arr_len)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "token slice [offset, offset + nTokens) exceeds array length");
        return ARGUS_DECODE_ERROR;
    }

    int32_t stack_buf[512];
    std::vector<int32_t> heap_buf;
    int32_t * tok_ptr = stack_buf;
    if (static_cast<size_t>(nTokens) > (sizeof(stack_buf) / sizeof(stack_buf[0]))) {
        try {
            heap_buf.resize(static_cast<size_t>(nTokens));
            tok_ptr = heap_buf.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate decode token buffer");
            return ARGUS_DECODE_ERROR;
        }
    }

    env->GetIntArrayRegion(tokens, offset, nTokens, reinterpret_cast<jint *>(tok_ptr));
    if (env->ExceptionCheck()) {
        return ARGUS_DECODE_ERROR;
    }

    return static_cast<jint>(argus_decode_tokens(
        ctx,
        tok_ptr,
        static_cast<int32_t>(nTokens),
        static_cast<int32_t>(startPos),
        static_cast<int32_t>(seqId),
        requestLogits == JNI_TRUE,
        reinterpret_cast<argus_abort_flag_t *>(abortFlagHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeDecodeTokensDirectBuffer(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jobject directBuf,
    jint byteOffset,
    jint nTokens,
    jint startPos,
    jint seqId,
    jboolean requestLogits,
    jlong abortFlagHandle) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !directBuf || byteOffset < 0 || (byteOffset & 3) != 0 || nTokens <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer decode arguments or unaligned byteOffset");
        return ARGUS_DECODE_ERROR;
    }
    auto * base = static_cast<const uint8_t *>(env->GetDirectBufferAddress(directBuf));
    const jlong cap = env->GetDirectBufferCapacity(directBuf);
    if (!base || cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "ByteBuffer is not a valid direct buffer");
        return ARGUS_DECODE_ERROR;
    }
    const int64_t required_end = static_cast<int64_t>(byteOffset) + static_cast<int64_t>(nTokens) * 4LL;
    if (required_end > static_cast<int64_t>(cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer token range exceeds capacity");
        return ARGUS_DECODE_ERROR;
    }
    const auto * tok_ptr = reinterpret_cast<const int32_t *>(base + byteOffset);
    return static_cast<jint>(argus_decode_tokens(
        ctx,
        tok_ptr,
        static_cast<int32_t>(nTokens),
        static_cast<int32_t>(startPos),
        static_cast<int32_t>(seqId),
        requestLogits == JNI_TRUE,
        reinterpret_cast<argus_abort_flag_t *>(abortFlagHandle)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeGetEmbeddings(
    JNIEnv * env, jclass /*cls*/, jlong ctxHandle, jint seqId, jfloatArray outEmbeddings, jint maxFloats) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !outEmbeddings || maxFloats <= 0 || env->GetArrayLength(outEmbeddings) < maxFloats) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid getEmbeddings arguments");
        return -1;
    }

    float stack_buf[1024];
    std::vector<float> heap_buf;
    float * ptr = stack_buf;
    if (static_cast<size_t>(maxFloats) > (sizeof(stack_buf) / sizeof(stack_buf[0]))) {
        try {
            heap_buf.resize(static_cast<size_t>(maxFloats));
            ptr = heap_buf.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate embeddings buffer");
            return -1;
        }
    }

    int32_t written = argus_get_embeddings(ctx, static_cast<int32_t>(seqId), ptr, static_cast<int32_t>(maxFloats));
    if (written > 0) {
        env->SetFloatArrayRegion(outEmbeddings, 0, static_cast<jsize>(written), ptr);
    }
    return static_cast<jint>(written);
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeGetEmbeddingsDirectBuffer(
    JNIEnv * env, jclass /*cls*/, jlong ctxHandle, jint seqId, jobject directBuf, jint byteOffset, jint maxFloats) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !directBuf || byteOffset < 0 || (byteOffset & 3) != 0 || maxFloats <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer getEmbeddings arguments");
        return -1;
    }
    auto * base = static_cast<uint8_t *>(env->GetDirectBufferAddress(directBuf));
    const jlong cap = env->GetDirectBufferCapacity(directBuf);
    if (!base || cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "ByteBuffer is not a valid direct buffer");
        return -1;
    }
    const int64_t required_end = static_cast<int64_t>(byteOffset) + static_cast<int64_t>(maxFloats) * 4LL;
    if (required_end > static_cast<int64_t>(cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer float range exceeds capacity");
        return -1;
    }
    auto * out_ptr = reinterpret_cast<float *>(base + byteOffset);
    return static_cast<jint>(argus_get_embeddings(ctx, static_cast<int32_t>(seqId), out_ptr, static_cast<int32_t>(maxFloats)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeEvalEmbeddings(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jintArray tokens,
    jint offset,
    jint nTokens,
    jint seqId,
    jfloatArray outEmbeddings,
    jint maxFloats) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !tokens || offset < 0 || nTokens <= 0 || !outEmbeddings || maxFloats <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid evalEmbeddings arguments");
        return -1;
    }
    if (static_cast<int64_t>(offset) + static_cast<int64_t>(nTokens) > static_cast<int64_t>(env->GetArrayLength(tokens)) ||
        env->GetArrayLength(outEmbeddings) < maxFloats) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "evalEmbeddings input or output array out of bounds");
        return -1;
    }

    int32_t stack_tok[512];
    std::vector<int32_t> heap_tok;
    int32_t * tok_ptr = stack_tok;
    if (static_cast<size_t>(nTokens) > (sizeof(stack_tok) / sizeof(stack_tok[0]))) {
        try {
            heap_tok.resize(static_cast<size_t>(nTokens));
            tok_ptr = heap_tok.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate evalEmbeddings token buffer");
            return -1;
        }
    }
    env->GetIntArrayRegion(tokens, offset, nTokens, reinterpret_cast<jint *>(tok_ptr));
    if (env->ExceptionCheck()) {
        return -1;
    }

    float stack_embd[1024];
    std::vector<float> heap_embd;
    float * embd_ptr = stack_embd;
    if (static_cast<size_t>(maxFloats) > (sizeof(stack_embd) / sizeof(stack_embd[0]))) {
        try {
            heap_embd.resize(static_cast<size_t>(maxFloats));
            embd_ptr = heap_embd.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate evalEmbeddings output buffer");
            return -1;
        }
    }

    int32_t written = argus_eval_embeddings(
        ctx,
        tok_ptr,
        static_cast<int32_t>(nTokens),
        static_cast<int32_t>(seqId),
        embd_ptr,
        static_cast<int32_t>(maxFloats));
    if (written > 0) {
        env->SetFloatArrayRegion(outEmbeddings, 0, static_cast<jsize>(written), embd_ptr);
    }
    return static_cast<jint>(written);
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeEvalEmbeddingsDirectBuffer(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jobject tokensBuf,
    jint tokensByteOffset,
    jint nTokens,
    jint seqId,
    jobject outEmbeddingsBuf,
    jint outByteOffset,
    jint maxFloats) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || !tokensBuf || !outEmbeddingsBuf ||
        tokensByteOffset < 0 || (tokensByteOffset & 3) != 0 || nTokens <= 0 ||
        outByteOffset < 0 || (outByteOffset & 3) != 0 || maxFloats <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer evalEmbeddings arguments");
        return -1;
    }
    const auto * tok_base = static_cast<const uint8_t *>(env->GetDirectBufferAddress(tokensBuf));
    const jlong tok_cap = env->GetDirectBufferCapacity(tokensBuf);
    auto * out_base = static_cast<uint8_t *>(env->GetDirectBufferAddress(outEmbeddingsBuf));
    const jlong out_cap = env->GetDirectBufferCapacity(outEmbeddingsBuf);
    if (!tok_base || tok_cap < 0 || !out_base || out_cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "tokensBuf or outEmbeddingsBuf is not a valid direct buffer");
        return -1;
    }
    const int64_t tok_end = static_cast<int64_t>(tokensByteOffset) + static_cast<int64_t>(nTokens) * 4LL;
    const int64_t out_end = static_cast<int64_t>(outByteOffset) + static_cast<int64_t>(maxFloats) * 4LL;
    if (tok_end > static_cast<int64_t>(tok_cap) || out_end > static_cast<int64_t>(out_cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer range exceeds capacity in evalEmbeddings");
        return -1;
    }
    const auto * tok_ptr = reinterpret_cast<const int32_t *>(tok_base + tokensByteOffset);
    auto * out_ptr = reinterpret_cast<float *>(out_base + outByteOffset);
    return static_cast<jint>(argus_eval_embeddings(
        ctx,
        tok_ptr,
        static_cast<int32_t>(nTokens),
        static_cast<int32_t>(seqId),
        out_ptr,
        static_cast<int32_t>(maxFloats)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSampleToken(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId, jfloat temperature, jfloat repeatPenalty) {
    return static_cast<jint>(argus_sample_token(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        static_cast<float>(temperature),
        static_cast<float>(repeatPenalty)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSampleTokenWithBias(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jint seqId,
    jfloat temperature,
    jfloat repeatPenalty,
    jintArray biasTokens,
    jfloatArray biasValues,
    jint biasCount) {
    std::vector<argus_logit_bias_t> biases;
    if (!copy_logit_biases(env, biasTokens, biasValues, biasCount, biases)) {
        return -1;
    }
    return static_cast<jint>(argus_sample_token_with_bias(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        static_cast<float>(temperature),
        static_cast<float>(repeatPenalty),
        biases.empty() ? nullptr : biases.data(),
        static_cast<int32_t>(biases.size())));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSampleTokenExt(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jint seqId,
    jfloat temperature,
    jfloat repeatPenalty,
    jint repeatLastN,
    jfloat frequencyPenalty,
    jfloat presencePenalty,
    jfloat topP,
    jfloat minP,
    jint topK,
    jfloat dryMultiplier,
    jfloat dryBase,
    jint dryAllowedLength,
    jint dryPenaltyLastN,
    jint seed,
    jintArray biasTokens,
    jfloatArray biasValues,
    jint biasCount) {
    std::vector<argus_logit_bias_t> biases;
    if (!copy_logit_biases(env, biasTokens, biasValues, biasCount, biases)) {
        return -1;
    }
    argus_sampler_params_t sparams = {};
    sparams.temperature        = static_cast<float>(temperature);
    sparams.repeat_penalty     = static_cast<float>(repeatPenalty);
    sparams.repeat_last_n      = static_cast<int32_t>(repeatLastN);
    sparams.frequency_penalty  = static_cast<float>(frequencyPenalty);
    sparams.presence_penalty   = static_cast<float>(presencePenalty);
    sparams.top_p              = static_cast<float>(topP);
    sparams.min_p              = static_cast<float>(minP);
    sparams.top_k              = static_cast<int32_t>(topK);
    sparams.dry_multiplier     = static_cast<float>(dryMultiplier);
    sparams.dry_base           = static_cast<float>(dryBase);
    sparams.dry_allowed_length = static_cast<int32_t>(dryAllowedLength);
    sparams.dry_penalty_last_n = static_cast<int32_t>(dryPenaltyLastN);
    sparams.seed               = static_cast<uint32_t>(seed);

    return static_cast<jint>(argus_sample_token_ext(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        &sparams,
        biases.empty() ? nullptr : biases.data(),
        static_cast<int32_t>(biases.size())));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerReset(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_sampler_reset(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerPrime(
    JNIEnv * env, jclass /*cls*/, jlong ctxHandle, jint seqId, jintArray tokens, jint offset, jint nTokens) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    if (!ctx || offset < 0 || nTokens < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid samplerPrime arguments");
        return -1;
    }
    if (nTokens == 0) {
        return static_cast<jint>(argus_sampler_prime(ctx, static_cast<int32_t>(seqId), nullptr, 0));
    }
    if (!tokens || static_cast<int64_t>(offset) + static_cast<int64_t>(nTokens) > static_cast<int64_t>(env->GetArrayLength(tokens))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "samplerPrime token slice out of bounds");
        return -1;
    }

    int32_t stack_buf[512];
    std::vector<int32_t> heap_buf;
    int32_t * tok_ptr = stack_buf;
    if (static_cast<size_t>(nTokens) > (sizeof(stack_buf) / sizeof(stack_buf[0]))) {
        try {
            heap_buf.resize(static_cast<size_t>(nTokens));
            tok_ptr = heap_buf.data();
        } catch (...) {
            set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate samplerPrime buffer");
            return -1;
        }
    }
    env->GetIntArrayRegion(tokens, offset, nTokens, reinterpret_cast<jint *>(tok_ptr));
    if (env->ExceptionCheck()) {
        return -1;
    }
    return static_cast<jint>(argus_sampler_prime(ctx, static_cast<int32_t>(seqId), tok_ptr, static_cast<int32_t>(nTokens)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerTruncate(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId, jint newLength) {
    return static_cast<jint>(argus_sampler_truncate(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        static_cast<int32_t>(newLength)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerGetHistoryCount(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_sampler_get_history_count(
        reinterpret_cast<const argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerHasPending(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_sampler_has_pending(
        reinterpret_cast<const argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSamplerDiscardPending(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_sampler_discard_pending(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheClearSlot(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId, jint p0, jint p1) {
    return argus_kv_cache_clear_slot(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        static_cast<int32_t>(p0),
        static_cast<int32_t>(p1)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheCanShift(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle) {
    return argus_kv_cache_can_shift(reinterpret_cast<const argus_context_t *>(ctxHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheSeqAdd(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId, jint p0, jint p1, jint delta) {
    return argus_kv_cache_seq_add(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId),
        static_cast<int32_t>(p0),
        static_cast<int32_t>(p1),
        static_cast<int32_t>(delta)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheSeqCp(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqIdSrc, jint seqIdDst, jint p0, jint p1) {
    return argus_kv_cache_seq_cp(
        reinterpret_cast<argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqIdSrc),
        static_cast<int32_t>(seqIdDst),
        static_cast<int32_t>(p0),
        static_cast<int32_t>(p1)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheSeqPosMax(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_kv_cache_seq_pos_max(
        reinterpret_cast<const argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeKvCacheSeqPosMin(
    JNIEnv * /*env*/, jclass /*cls*/, jlong ctxHandle, jint seqId) {
    return static_cast<jint>(argus_kv_cache_seq_pos_min(
        reinterpret_cast<const argus_context_t *>(ctxHandle),
        static_cast<int32_t>(seqId)));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeSynthesizeSpeech(
    JNIEnv * env, jclass /*cls*/,
    jlong ctxHandle,
    jlong vocoderModelHandle,
    jbyteArray utf8Text,
    jint voiceSeed,
    jfloatArray outPcm,
    jint maxSamples) {
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    const auto * vocoder = reinterpret_cast<const argus_model_t *>(vocoderModelHandle);
    if (!ctx || !vocoder || !utf8Text || !outPcm || maxSamples <= 0 || env->GetArrayLength(outPcm) < maxSamples) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid synthesizeSpeech arguments");
        return -1;
    }
    utf8_span text(env, utf8Text, true);
    if (!text.valid) {
        return -1;
    }
    try {
        std::vector<float> pcm_buf(static_cast<size_t>(maxSamples));
        int32_t written = argus_synthesize_speech_n(
            ctx,
            vocoder,
            text.ptr,
            text.len,
            static_cast<int32_t>(voiceSeed),
            pcm_buf.data(),
            static_cast<int32_t>(maxSamples),
            nullptr,
            0);
        if (written > 0) {
            const jsize to_copy = static_cast<jsize>(written <= maxSamples ? written : maxSamples);
            env->SetFloatArrayRegion(outPcm, 0, to_copy, pcm_buf.data());
        }
        return static_cast<jint>(written);
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate synthesizeSpeech PCM buffer");
        return -1;
    }
}

// =========================================================================
// 4. Atomic Cancellation Controls (ArgusAbortFlag)
// =========================================================================

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagCreate(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return reinterpret_cast<jlong>(argus_abort_flag_create());
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagRequest(
    JNIEnv * /*env*/, jclass /*cls*/, jlong flagHandle) {
    argus_abort_flag_request(reinterpret_cast<argus_abort_flag_t *>(flagHandle));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagReset(
    JNIEnv * /*env*/, jclass /*cls*/, jlong flagHandle) {
    argus_abort_flag_reset(reinterpret_cast<argus_abort_flag_t *>(flagHandle));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagIsRequested(
    JNIEnv * /*env*/, jclass /*cls*/, jlong flagHandle) {
    return argus_abort_flag_is_requested(reinterpret_cast<const argus_abort_flag_t *>(flagHandle))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagRetain(
    JNIEnv * /*env*/, jclass /*cls*/, jlong flagHandle) {
    return argus_abort_flag_retain(reinterpret_cast<argus_abort_flag_t *>(flagHandle))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAbortFlagRelease(
    JNIEnv * /*env*/, jclass /*cls*/, jlong flagHandle) {
    argus_abort_flag_release(reinterpret_cast<argus_abort_flag_t *>(flagHandle));
}

// =========================================================================
// 5. Audio Stream Transcription Subsystem (Whisper ASR)
// =========================================================================

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAudioInit(
    JNIEnv * env, jclass /*cls*/, jbyteArray whisperModelPathUtf8, jint cpuThreads, jint gpuLayers) {
    if (!whisperModelPathUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "whisper model path is null");
        return 0L;
    }
    utf8_span path(env, whisperModelPathUtf8, true);
    if (!path.valid) {
        return 0L;
    }
    argus_audio_params_t params = {};
    params.whisper_model_path = path.ptr;
    params.cpu_threads        = static_cast<int32_t>(cpuThreads);
    params.gpu_layers         = static_cast<int32_t>(gpuLayers);
    return reinterpret_cast<jlong>(argus_audio_init(&params));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAudioFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong audioCtxHandle) {
    argus_audio_free(reinterpret_cast<argus_audio_context_t *>(audioCtxHandle));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAudioSetNThreads(
    JNIEnv * /*env*/, jclass /*cls*/, jlong audioCtxHandle, jint nThreads) {
    argus_audio_set_n_threads(reinterpret_cast<argus_audio_context_t *>(audioCtxHandle), static_cast<int32_t>(nThreads));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeAudioGetNThreads(
    JNIEnv * /*env*/, jclass /*cls*/, jlong audioCtxHandle) {
    return static_cast<jint>(argus_audio_get_n_threads(reinterpret_cast<argus_audio_context_t *>(audioCtxHandle)));
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeTranscribeAudio(
    JNIEnv * env, jclass /*cls*/,
    jlong audioCtxHandle,
    jfloatArray pcmData,
    jint offset,
    jint sampleCount,
    jint maxChars) {
    auto * ctx = reinterpret_cast<argus_audio_context_t *>(audioCtxHandle);
    if (!ctx || !pcmData || offset < 0 || sampleCount <= 0 || maxChars <= 0 ||
        maxChars == std::numeric_limits<int32_t>::max()) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid transcribeAudio parameters");
        return nullptr;
    }
    if (static_cast<int64_t>(offset) + static_cast<int64_t>(sampleCount) > static_cast<int64_t>(env->GetArrayLength(pcmData))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "PCM slice [offset, offset + sampleCount) exceeds array length");
        return nullptr;
    }
    try {
        std::vector<float> pcm_buf(static_cast<size_t>(sampleCount));
        env->GetFloatArrayRegion(pcmData, offset, sampleCount, pcm_buf.data());
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        std::vector<char> out_text(static_cast<size_t>(maxChars) + 1);
        int32_t written = argus_transcribe_audio(
            ctx,
            pcm_buf.data(),
            static_cast<int32_t>(sampleCount),
            out_text.data(),
            static_cast<int32_t>(out_text.size()));
        if (written < 0) {
            return nullptr;
        }
        return bytes_from_utf8(env, out_text.data(), static_cast<size_t>(written));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate transcription buffers");
        return nullptr;
    }
}

ARGUS_JNI_EXPORT jbyteArray JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeTranscribeAudioDirectBuffer(
    JNIEnv * env, jclass /*cls*/,
    jlong audioCtxHandle,
    jobject pcmBuf,
    jint byteOffset,
    jint sampleCount,
    jint maxChars) {
    auto * ctx = reinterpret_cast<argus_audio_context_t *>(audioCtxHandle);
    if (!ctx || !pcmBuf || byteOffset < 0 || (byteOffset & 3) != 0 || sampleCount <= 0 ||
        maxChars <= 0 || maxChars == std::numeric_limits<int32_t>::max()) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer transcribeAudio parameters");
        return nullptr;
    }
    const auto * base = static_cast<const uint8_t *>(env->GetDirectBufferAddress(pcmBuf));
    const jlong cap = env->GetDirectBufferCapacity(pcmBuf);
    if (!base || cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "pcmBuf is not a valid direct buffer");
        return nullptr;
    }
    const int64_t required_end = static_cast<int64_t>(byteOffset) + static_cast<int64_t>(sampleCount) * 4LL;
    if (required_end > static_cast<int64_t>(cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer PCM range exceeds capacity");
        return nullptr;
    }
    try {
        std::vector<char> out_text(static_cast<size_t>(maxChars) + 1);
        const auto * pcm_ptr = reinterpret_cast<const float *>(base + byteOffset);
        int32_t written = argus_transcribe_audio(
            ctx,
            pcm_ptr,
            static_cast<int32_t>(sampleCount),
            out_text.data(),
            static_cast<int32_t>(out_text.size()));
        if (written < 0) {
            return nullptr;
        }
        return bytes_from_utf8(env, out_text.data(), static_cast<size_t>(written));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate transcription output buffer");
        return nullptr;
    }
}

// =========================================================================
// 6. Multimodal Vision / Audio / Video Operations
// =========================================================================

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalInit(
    JNIEnv * env, jclass /*cls*/, jlong modelHandle, jbyteArray mmprojPathUtf8, jint cpuThreads, jboolean useGpu) {
    const auto * model = reinterpret_cast<const argus_model_t *>(modelHandle);
    if (!model || !mmprojPathUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "null model or mmproj path");
        return 0L;
    }
    utf8_span path(env, mmprojPathUtf8, true);
    if (!path.valid) {
        return 0L;
    }
    argus_multimodal_params_t params = {};
    params.mmproj_path = path.ptr;
    params.cpu_threads = static_cast<int32_t>(cpuThreads);
    params.use_gpu     = (useGpu == JNI_TRUE);
    return reinterpret_cast<jlong>(argus_multimodal_init(model, &params));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalRetain(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    return argus_multimodal_retain(reinterpret_cast<argus_multimodal_t *>(mctxHandle)) ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalRelease(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    argus_multimodal_release(reinterpret_cast<argus_multimodal_t *>(mctxHandle));
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    argus_multimodal_free(reinterpret_cast<argus_multimodal_t *>(mctxHandle));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalSupportVision(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    return argus_multimodal_support_vision(reinterpret_cast<const argus_multimodal_t *>(mctxHandle))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalSupportAudio(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    return argus_multimodal_support_audio(reinterpret_cast<const argus_multimodal_t *>(mctxHandle))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalSupportVideo(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    return argus_multimodal_support_video(reinterpret_cast<const argus_multimodal_t *>(mctxHandle))
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalGetAudioSampleRate(
    JNIEnv * /*env*/, jclass /*cls*/, jlong mctxHandle) {
    return static_cast<jint>(argus_multimodal_get_audio_sample_rate(
        reinterpret_cast<const argus_multimodal_t *>(mctxHandle)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapFromRgb(
    JNIEnv * env, jclass /*cls*/, jint width, jint height, jbyteArray rgbData) {
    if (!rgbData || width <= 0 || height <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid RGB dimensions or null rgbData");
        return 0L;
    }
    const int64_t expected_bytes = static_cast<int64_t>(width) * static_cast<int64_t>(height) * 3LL;
    if (expected_bytes <= 0 ||
        expected_bytes > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
        expected_bytes > static_cast<int64_t>(env->GetArrayLength(rgbData))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "rgbData array smaller than width * height * 3 or exceeds int32");
        return 0L;
    }
    try {
        std::vector<uint8_t> buf(static_cast<size_t>(expected_bytes));
        env->GetByteArrayRegion(rgbData, 0, static_cast<jsize>(expected_bytes), reinterpret_cast<jbyte *>(buf.data()));
        if (env->ExceptionCheck()) {
            return 0L;
        }
        return reinterpret_cast<jlong>(argus_bitmap_from_rgb(
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            buf.data()));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate RGB bitmap staging buffer");
        return 0L;
    }
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapFromRgbDirectBuffer(
    JNIEnv * env, jclass /*cls*/, jint width, jint height, jobject rgbBuf, jint byteOffset) {
    if (!rgbBuf || width <= 0 || height <= 0 || byteOffset < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer RGB bitmap arguments");
        return 0L;
    }
    const auto * base = static_cast<const uint8_t *>(env->GetDirectBufferAddress(rgbBuf));
    const jlong cap = env->GetDirectBufferCapacity(rgbBuf);
    if (!base || cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "rgbBuf is not a valid direct buffer");
        return 0L;
    }
    const int64_t expected_bytes = static_cast<int64_t>(width) * static_cast<int64_t>(height) * 3LL;
    if (expected_bytes <= 0 || static_cast<int64_t>(byteOffset) + expected_bytes > static_cast<int64_t>(cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer RGB data smaller than width * height * 3");
        return 0L;
    }
    return reinterpret_cast<jlong>(argus_bitmap_from_rgb(
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        base + byteOffset));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapFromPcm(
    JNIEnv * env, jclass /*cls*/, jfloatArray pcmData, jint nSamples) {
    if (!pcmData || nSamples <= 0 || env->GetArrayLength(pcmData) < nSamples) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid PCM bitmap arguments");
        return 0L;
    }
    try {
        std::vector<float> buf(static_cast<size_t>(nSamples));
        env->GetFloatArrayRegion(pcmData, 0, nSamples, buf.data());
        if (env->ExceptionCheck()) {
            return 0L;
        }
        return reinterpret_cast<jlong>(argus_bitmap_from_pcm(buf.data(), static_cast<int32_t>(nSamples)));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate PCM bitmap staging buffer");
        return 0L;
    }
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapFromPcmDirectBuffer(
    JNIEnv * env, jclass /*cls*/, jobject pcmBuf, jint byteOffset, jint nSamples) {
    if (!pcmBuf || byteOffset < 0 || (byteOffset & 3) != 0 || nSamples <= 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid direct buffer PCM bitmap arguments");
        return 0L;
    }
    const auto * base = static_cast<const uint8_t *>(env->GetDirectBufferAddress(pcmBuf));
    const jlong cap = env->GetDirectBufferCapacity(pcmBuf);
    if (!base || cap < 0) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "pcmBuf is not a valid direct buffer");
        return 0L;
    }
    const int64_t required_end = static_cast<int64_t>(byteOffset) + static_cast<int64_t>(nSamples) * 4LL;
    if (required_end > static_cast<int64_t>(cap)) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "direct buffer PCM data smaller than nSamples * 4");
        return 0L;
    }
    return reinterpret_cast<jlong>(argus_bitmap_from_pcm(
        reinterpret_cast<const float *>(base + byteOffset),
        static_cast<int32_t>(nSamples)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapLoadFile(
    JNIEnv * env, jclass /*cls*/, jlong mctxHandle, jbyteArray pathUtf8, jboolean placeholder) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    if (!mctx || !pathUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid multimodal handle or file path");
        return 0L;
    }
    utf8_span path(env, pathUtf8, true);
    if (!path.valid) {
        return 0L;
    }
    return reinterpret_cast<jlong>(argus_bitmap_load_file(mctx, path.ptr, placeholder == JNI_TRUE));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapLoadBuffer(
    JNIEnv * env, jclass /*cls*/, jlong mctxHandle, jbyteArray buffer, jint offset, jint size, jboolean placeholder) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    if (!mctx || !buffer || offset < 0 || size <= 0 ||
        static_cast<int64_t>(offset) + static_cast<int64_t>(size) > static_cast<int64_t>(env->GetArrayLength(buffer))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid bitmapLoadBuffer arguments");
        return 0L;
    }
    try {
        std::vector<uint8_t> buf(static_cast<size_t>(size));
        env->GetByteArrayRegion(buffer, offset, size, reinterpret_cast<jbyte *>(buf.data()));
        if (env->ExceptionCheck()) {
            return 0L;
        }
        return reinterpret_cast<jlong>(argus_bitmap_load_buffer(
            mctx,
            buf.data(),
            static_cast<int32_t>(size),
            placeholder == JNI_TRUE));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate bitmap buffer");
        return 0L;
    }
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong bitmapHandle) {
    argus_bitmap_free(reinterpret_cast<argus_bitmap_t *>(bitmapHandle));
}

ARGUS_JNI_EXPORT jboolean JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeBitmapSetMergeable(
    JNIEnv * /*env*/, jclass /*cls*/, jlong bitmapHandle, jboolean mergeable) {
    return argus_bitmap_set_mergeable(reinterpret_cast<argus_bitmap_t *>(bitmapHandle), mergeable == JNI_TRUE)
        ? JNI_TRUE : JNI_FALSE;
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVideoLoadFile(
    JNIEnv * env, jclass /*cls*/, jlong mctxHandle, jbyteArray pathUtf8, jfloat fpsTarget, jlong timestampIntervalMs) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    if (!mctx || !pathUtf8) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid multimodal handle or video file path");
        return 0L;
    }
    utf8_span path(env, pathUtf8, true);
    if (!path.valid) {
        return 0L;
    }
    return reinterpret_cast<jlong>(argus_video_load_file(
        mctx,
        path.ptr,
        static_cast<float>(fpsTarget),
        static_cast<int64_t>(timestampIntervalMs)));
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVideoLoadBuffer(
    JNIEnv * env, jclass /*cls*/,
    jlong mctxHandle,
    jbyteArray buffer,
    jint offset,
    jint size,
    jfloat fpsTarget,
    jlong timestampIntervalMs) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    if (!mctx || !buffer || offset < 0 || size <= 0 ||
        static_cast<int64_t>(offset) + static_cast<int64_t>(size) > static_cast<int64_t>(env->GetArrayLength(buffer))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid videoLoadBuffer arguments");
        return 0L;
    }
    try {
        std::vector<uint8_t> buf(static_cast<size_t>(size));
        env->GetByteArrayRegion(buffer, offset, size, reinterpret_cast<jbyte *>(buf.data()));
        if (env->ExceptionCheck()) {
            return 0L;
        }
        return reinterpret_cast<jlong>(argus_video_load_buffer(
            mctx,
            buf.data(),
            static_cast<int32_t>(size),
            static_cast<float>(fpsTarget),
            static_cast<int64_t>(timestampIntervalMs)));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate video staging buffer");
        return 0L;
    }
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVideoFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong videoHandle) {
    argus_video_free(reinterpret_cast<argus_video_t *>(videoHandle));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeVideoReadNext(
    JNIEnv * env, jclass /*cls*/, jlong videoHandle, jlongArray outBitmapHandle, jobjectArray outTextBytes) {
    auto * video = reinterpret_cast<argus_video_t *>(videoHandle);
    if (!video || !outBitmapHandle || !outTextBytes ||
        env->GetArrayLength(outBitmapHandle) < 1 || env->GetArrayLength(outTextBytes) < 1) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid videoReadNext arguments");
        return -2;
    }
    argus_bitmap_t * bmp = nullptr;
    char text_buf[1024] = {0};
    int32_t res = argus_video_read_next(video, &bmp, text_buf, static_cast<int32_t>(sizeof(text_buf)));
    jlong bmp_handle = reinterpret_cast<jlong>(bmp);
    env->SetLongArrayRegion(outBitmapHandle, 0, 1, &bmp_handle);
    if (res == 0 && !bmp && text_buf[0] != '\0') {
        jbyteArray utf8_bytes = bytes_from_utf8(env, text_buf, std::strlen(text_buf));
        env->SetObjectArrayElement(outTextBytes, 0, utf8_bytes);
    } else {
        env->SetObjectArrayElement(outTextBytes, 0, nullptr);
    }
    return static_cast<jint>(res);
}

ARGUS_JNI_EXPORT jlong JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeInputChunksInit(
    JNIEnv * /*env*/, jclass /*cls*/) {
    return reinterpret_cast<jlong>(argus_input_chunks_init());
}

ARGUS_JNI_EXPORT void JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeInputChunksFree(
    JNIEnv * /*env*/, jclass /*cls*/, jlong chunksHandle) {
    argus_input_chunks_free(reinterpret_cast<argus_input_chunks_t *>(chunksHandle));
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeMultimodalTokenize(
    JNIEnv * env, jclass /*cls*/,
    jlong mctxHandle,
    jlong chunksHandle,
    jbyteArray utf8Text,
    jboolean addBos,
    jlongArray bitmapHandles,
    jint nBitmaps) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    auto * chunks = reinterpret_cast<argus_input_chunks_t *>(chunksHandle);
    if (!mctx || !chunks || !utf8Text || nBitmaps < 0 ||
        (nBitmaps > 0 && (!bitmapHandles || env->GetArrayLength(bitmapHandles) < nBitmaps))) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid multimodalTokenize arguments");
        return -1;
    }
    utf8_span text(env, utf8Text, true);
    if (!text.valid) {
        return -1;
    }
    try {
        std::vector<const argus_bitmap_t *> bitmaps(static_cast<size_t>(nBitmaps));
        if (nBitmaps > 0) {
            std::vector<jlong> handles(static_cast<size_t>(nBitmaps));
            env->GetLongArrayRegion(bitmapHandles, 0, nBitmaps, handles.data());
            if (env->ExceptionCheck()) {
                return -1;
            }
            for (jint i = 0; i < nBitmaps; ++i) {
                bitmaps[static_cast<size_t>(i)] = reinterpret_cast<const argus_bitmap_t *>(handles[static_cast<size_t>(i)]);
            }
        }
        return static_cast<jint>(argus_multimodal_tokenize_n(
            mctx,
            chunks,
            text.ptr,
            text.len,
            addBos == JNI_TRUE,
            bitmaps.empty() ? nullptr : bitmaps.data(),
            static_cast<int32_t>(nBitmaps)));
    } catch (...) {
        set_last_error(ARGUS_ERROR_OUT_OF_MEMORY, "failed to allocate multimodal bitmap pointer array");
        return -1;
    }
}

ARGUS_JNI_EXPORT jint JNICALL Java_cc_projectargus_libargus_internal_ArgusJni_nativeEvalMultimodalChunks(
    JNIEnv * env, jclass /*cls*/,
    jlong mctxHandle,
    jlong ctxHandle,
    jlong chunksHandle,
    jint nPast,
    jint seqId,
    jint nBatch,
    jboolean logitsLast,
    jintArray outNewNPast) {
    auto * mctx = reinterpret_cast<argus_multimodal_t *>(mctxHandle);
    auto * ctx = reinterpret_cast<argus_context_t *>(ctxHandle);
    const auto * chunks = reinterpret_cast<const argus_input_chunks_t *>(chunksHandle);
    if (!mctx || !ctx || !chunks || !outNewNPast || env->GetArrayLength(outNewNPast) < 1) {
        set_last_error(ARGUS_ERROR_INVALID_ARGUMENT, "invalid evalMultimodalChunks arguments");
        return -1;
    }
    int32_t new_n_past = 0;
    int32_t res = argus_eval_multimodal_chunks(
        mctx,
        ctx,
        chunks,
        static_cast<int32_t>(nPast),
        static_cast<int32_t>(seqId),
        static_cast<int32_t>(nBatch),
        logitsLast == JNI_TRUE,
        &new_n_past);
    if (res == 0) {
        jint val = static_cast<jint>(new_n_past);
        env->SetIntArrayRegion(outNewNPast, 0, 1, &val);
    }
    return static_cast<jint>(res);
}

} // extern "C"
