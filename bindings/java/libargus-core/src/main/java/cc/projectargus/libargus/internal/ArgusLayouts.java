package cc.projectargus.libargus.internal;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Memory layouts of all public C structures exposed by libargus.h.
 * These match 1:1 with the compiler alignment rules and explicit padding offsets.
 */
public final class ArgusLayouts {
    private ArgusLayouts() {}

    /**
     * Memory layout for argus_model_params_t
     * <pre>
     * typedef struct argus_model_params {
     *     const char * model_path;              // 8 bytes
     *     int32_t      gpu_layers;              // 4 bytes
     *     bool         use_mlock;               // 1 byte
     *     uint8_t      reserved_padding[3];     // 3 bytes padding
     * } argus_model_params_t;
     * </pre>
     */
    public static final StructLayout MODEL_PARAMS = MemoryLayout.structLayout(
        ValueLayout.ADDRESS.withName("model_path"),
        ValueLayout.JAVA_INT.withName("gpu_layers"),
        ValueLayout.JAVA_BOOLEAN.withName("use_mlock"),
        MemoryLayout.paddingLayout(3)
    ).withName("argus_model_params");

    /**
     * Memory layout for argus_context_params_t
     * <pre>
     * typedef struct argus_context_params {
     *     const argus_model_t * draft_model;     // 8 bytes
     *     int32_t               context_length;  // 4 bytes
     *     int32_t               cpu_threads;     // 4 bytes
     *     int32_t               type_k;          // 4 bytes
     *     int32_t               type_v;          // 4 bytes
     *     int32_t               spec_draft_n_max;// 4 bytes
     *     int32_t               u_batch;         // 4 bytes
     *     int32_t               n_seq_max;       // 4 bytes
     *     bool                  enable_draft_mtp;// 1 byte
     *     bool                  embeddings;      // 1 byte
     *     bool                  kv_unified;      // 1 byte
     *     uint8_t               reserved_padding[1];// 1 byte padding
     * } argus_context_params_t;
     * </pre>
     */
    public static final StructLayout CONTEXT_PARAMS = MemoryLayout.structLayout(
        ValueLayout.ADDRESS.withName("draft_model"),
        ValueLayout.JAVA_INT.withName("context_length"),
        ValueLayout.JAVA_INT.withName("cpu_threads"),
        ValueLayout.JAVA_INT.withName("type_k"),
        ValueLayout.JAVA_INT.withName("type_v"),
        ValueLayout.JAVA_INT.withName("spec_draft_n_max"),
        ValueLayout.JAVA_INT.withName("u_batch"),
        ValueLayout.JAVA_INT.withName("n_seq_max"),
        ValueLayout.JAVA_BOOLEAN.withName("enable_draft_mtp"),
        ValueLayout.JAVA_BOOLEAN.withName("embeddings"),
        ValueLayout.JAVA_BOOLEAN.withName("kv_unified"),
        MemoryLayout.paddingLayout(1)
    ).withName("argus_context_params");

    /**
     * Memory layout for argus_audio_params_t
     * <pre>
     * typedef struct argus_audio_params {
     *     const char * whisper_model_path;      // 8 bytes
     *     int32_t      cpu_threads;             // 4 bytes
     *     int32_t      gpu_layers;              // 4 bytes
     * } argus_audio_params_t;
     * </pre>
     */
    public static final StructLayout AUDIO_PARAMS = MemoryLayout.structLayout(
        ValueLayout.ADDRESS.withName("whisper_model_path"),
        ValueLayout.JAVA_INT.withName("cpu_threads"),
        ValueLayout.JAVA_INT.withName("gpu_layers")
    ).withName("argus_audio_params");

    /**
     * Memory layout for argus_token_batch_t
     * <pre>
     * typedef struct argus_token_batch {
     *     const int32_t * tokens;               // 8 bytes
     *     int32_t         n_tokens;             // 4 bytes
     *     int32_t         start_pos;            // 4 bytes
     *     int32_t         seq_id;               // 4 bytes
     *     bool            request_logits;       // 1 byte
     *     uint8_t         reserved_padding[3];  // 3 bytes padding
     *     const int32_t * abort_flag;           // 8 bytes pointer
     * } argus_token_batch_t;
     * </pre>
     */
    public static final StructLayout TOKEN_BATCH = MemoryLayout.structLayout(
        ValueLayout.ADDRESS.withName("tokens"),
        ValueLayout.JAVA_INT.withName("n_tokens"),
        ValueLayout.JAVA_INT.withName("start_pos"),
        ValueLayout.JAVA_INT.withName("seq_id"),
        ValueLayout.JAVA_BOOLEAN.withName("request_logits"),
        MemoryLayout.paddingLayout(3),
        ValueLayout.ADDRESS.withName("abort_flag")
    ).withName("argus_token_batch");

    /**
     * Memory layout for argus_multimodal_params_t
     * <pre>
     * typedef struct argus_multimodal_params {
     *     const char * mmproj_path;             // 8 bytes
     *     int32_t      cpu_threads;             // 4 bytes
     *     bool         use_gpu;                 // 1 byte
     *     uint8_t      reserved_padding[3];     // 3 bytes padding
     * } argus_multimodal_params_t;
     * </pre>
     */
    public static final StructLayout MULTIMODAL_PARAMS = MemoryLayout.structLayout(
        ValueLayout.ADDRESS.withName("mmproj_path"),
        ValueLayout.JAVA_INT.withName("cpu_threads"),
        ValueLayout.JAVA_BOOLEAN.withName("use_gpu"),
        MemoryLayout.paddingLayout(3)
    ).withName("argus_multimodal_params");

    /**
     * Memory layout for argus_logit_bias_t
     * <pre>
     * typedef struct argus_logit_bias {
     *     int32_t token;                        // 4 bytes
     *     float   bias;                         // 4 bytes
     * } argus_logit_bias_t;
     * </pre>
     */
    public static final StructLayout LOGIT_BIAS = MemoryLayout.structLayout(
        ValueLayout.JAVA_INT.withName("token"),
        ValueLayout.JAVA_FLOAT.withName("bias")
    ).withName("argus_logit_bias");
}
