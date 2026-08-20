/**
 * @file libargus.h
 * @brief Zero-allocation unified C API for Vision, Audio, Speech-to-Text, and LLM text generation.
 * @version 1.6.2
 * 
 * libargus provides an optimized, model-agnostic unmanaged orchestration layer over 
 * GGML compute primitives. This file defines a strict, flat C Application Binary 
 * Interface (ABI) featuring deterministic structural memory tracking for modern 
 * off-heap Java/JVM Project Panama binding frameworks.
 * 
 * Copyright (c) 2026 Project Argus Open-Source Core.
 */

#ifndef LIBARGUS_H
#define LIBARGUS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(ARGUS_BUILDING_DLL)
    #define ARGUS_API __declspec(dllexport)
  #else
    #define ARGUS_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) || defined(__clang__)
    #define ARGUS_API __attribute__((visibility("default")))
  #else
    #define ARGUS_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Opaque Structural Core Context Types
// =========================================================================

/**
 * @brief Opaque handler representing loaded GGUF model weights in memory.
 */
typedef struct argus_model argus_model_t;

/**
 * @brief Opaque handler representing an active execution session context.
 */
typedef struct argus_context argus_context_t;

/**
 * @brief Opaque handler representing the active Whisper ASR processing runtime.
 */
typedef struct argus_audio_context argus_audio_context_t;

// =========================================================================
// Memory-Aligned Configuration & Transaction Layouts
// =========================================================================

/**
 * @brief GGML tensor type representations for KV cache quantization settings.
 */
typedef enum argus_kv_type {
    ARGUS_KV_TYPE_F16  = 0,  /**< 16-bit float (default) */
    ARGUS_KV_TYPE_Q8_0 = 8,  /**< 8-bit quantization */
    ARGUS_KV_TYPE_Q4_0 = 2,  /**< 4-bit quantization (type 0) */
    ARGUS_KV_TYPE_Q4_1 = 3,  /**< 4-bit quantization (type 1) */
} argus_kv_type_t;

/**
 * @brief Parameters for configuring and loading the GGUF model weights.
 */
typedef struct argus_model_params {
    const char * model_path;              /**< Path to the target GGUF file (8 bytes) */
    int32_t      gpu_layers;              /**< Compute layers offloaded to VRAM (4 bytes) */
    bool         use_mlock;               /**< Force OS page locking (1 byte) */
    uint8_t      reserved_padding[3];     /**< Align struct layout boundary (3 bytes) */
} argus_model_params_t;

/**
 * @brief Parameters for configuring the execution session context.
 */
typedef struct argus_context_params {
    const argus_model_t * draft_model;     /**< Optional draft model for speculative decoding (8 bytes) */
    int32_t               context_length;  /**< Context allocation capacity (4 bytes) */
    int32_t               cpu_threads;     /**< Threads reserved for calculations (4 bytes) */
    int32_t               type_k;          /**< Key cache quantization format (argus_kv_type_t) (4 bytes) */
    int32_t               type_v;          /**< Value cache quantization format (argus_kv_type_t) (4 bytes) */
    int32_t               spec_draft_n_max;/**< Max speculative draft tokens to check (4 bytes) */
    int32_t               u_batch;         /**< Physical micro-batch size (0 = default/auto) (4 bytes) */
    int32_t               n_seq_max;       /**< Maximum sequence slots (0 = auto-detect) (4 bytes) */
    bool                  enable_draft_mtp;/**< Enable Multi-Token Prediction draft head (1 byte) */
    bool                  embeddings;      /**< Enable embeddings output (1 byte) */
    bool                  kv_unified;      /**< Dynamic KV cache sharing across sequences (1 byte) */
    uint8_t               reserved_padding[1];/**< Struct alignment padding (1 byte) */
} argus_context_params_t;

/**
 * @brief Parameters for configuring the Whisper speech-to-text decoder context.
 */
typedef struct argus_audio_params {
    const char * whisper_model_path;      /**< Path to the Whisper model file (8 bytes) */
    int32_t      cpu_threads;             /**< Thread allocation cap for acoustic math (4 bytes) */
    int32_t      gpu_layers;              /**< Total encoder/decoder blocks shifted to GPU space (4 bytes) */
} argus_audio_params_t;

/**
 * @brief Transactional payload metadata representing an evaluation turn batch step.
 */
typedef struct argus_token_batch {
    const int32_t * tokens;               /**< Pointer targeting an aligned array of token IDs (8 bytes) */
    int32_t         n_tokens;             /**< Token count to evaluate (4 bytes) */
    int32_t         start_pos;            /**< Offsets in KV cache (4 bytes) */
    int32_t         seq_id;               /**< Target sequence slot tracking ID (4 bytes) */
    bool            request_logits;       /**< Evaluate output logits on terminal slot (1 byte) */
    uint8_t         reserved_padding[3];  /**< Alignment padding securing 4-byte boundaries (3 bytes) */
    const int32_t * abort_flag;           /**< Optional pointer targeting a cancellation flag (8 bytes) */
} argus_token_batch_t;

/**
 * @brief Parameters for configuring the multimodal projector.
 */
typedef struct argus_multimodal_params {
    const char * mmproj_path;             /**< Path to the GGUF mmproj file (8 bytes) */
    int32_t      cpu_threads;             /**< Thread allocation cap for projection (4 bytes) */
    bool         use_gpu;                 /**< Enable GPU offloading of projector (1 byte) */
    uint8_t      reserved_padding[3];     /**< Align struct layout boundary (3 bytes) */
} argus_multimodal_params_t;

/**
 * @brief Representation of a single token logit steering bias.
 */
typedef struct argus_logit_bias {
    int32_t token;                        /**< Target vocabulary token ID (4 bytes) */
    float   bias;                         /**< Logit adjustment bias weight multiplier (4 bytes) */
} argus_logit_bias_t;

/**
 * @brief Extended sampling configuration parameters.
 */
typedef struct argus_sampler_params {
    float    temperature;          /**< Entropy control value (<= 0.0f = greedy) (4 bytes) */
    float    repeat_penalty;       /**< Repetition penalty factor (<= 1.0f = disabled) (4 bytes) */
    int32_t  repeat_last_n;        /**< Last n tokens to penalize (0 = default 64) (4 bytes) */
    float    frequency_penalty;    /**< Frequency penalty factor (0.0f = disabled) (4 bytes) */
    float    presence_penalty;     /**< Presence penalty factor (0.0f = disabled) (4 bytes) */
    float    top_p;                /**< Nucleus sampling threshold (>= 1.0f = disabled) (4 bytes) */
    float    min_p;                /**< Minimum probability threshold (<= 0.0f = disabled) (4 bytes) */
    int32_t  top_k;                /**< Top-K sampling cap (<= 0 = disabled) (4 bytes) */
    float    dry_multiplier;       /**< DRY repetition penalty multiplier (0.0f = disabled) (4 bytes) */
    float    dry_base;             /**< DRY exponential base (1.75f default) (4 bytes) */
    int32_t  dry_allowed_length;   /**< DRY allowed n-gram length before penalty (2 default) (4 bytes) */
    int32_t  dry_penalty_last_n;   /**< DRY penalty lookback window (-1 = full context) (4 bytes) */
    uint64_t seed;                 /**< RNG seed for distribution sampling (0xFFFFFFFFFFFFFFFFULL = random) (8 bytes) */
} argus_sampler_params_t;

// =========================================================================
// 1. Process-Global Subsystem Lifecycle Control
// =========================================================================

/**
 * @brief Initializes process-wide unmanaged backends and loads device drivers.
 * Must be executed exactly once per runtime lifespan prior to model instantiations.
 * @return true if compute nodes are successfully prepared, false otherwise.
 */
ARGUS_API bool argus_backend_init(const char * custom_plugin_path);

/**
 * @brief Retrieves the compiled library version.
 * @return Null-terminated project version string (e.g. "0.2.3").
 */
ARGUS_API const char * argus_version(void);

/**
 * @brief Deallocates global hardware device links and cleans process space.
 */
ARGUS_API void argus_backend_free(void);

/**
 * @brief Gets the total count of registered hardware acceleration backends.
 * @return Total backend count.
 */
ARGUS_API int32_t argus_backend_get_count(void);

/**
 * @brief Gets the name of a registered hardware acceleration backend.
 * @param index Registry index.
 * @return Null-terminated driver registry name string.
 */
ARGUS_API const char * argus_backend_get_name(int32_t index);

// =========================================================================
// 2. Model & Context Lifecycle Managers
// =========================================================================

/**
 * @brief Loads a GGUF model from the specified parameters.
 * @param params Configurations detailing file path, mlock, and GPU offloading.
 * @return Opaque model pointer, or NULL on failure.
 */
ARGUS_API argus_model_t * argus_model_load(const argus_model_params_t * params);

/**
 * @brief Frees loaded GGUF model resources.
 * @param model Reference model pointer.
 */
ARGUS_API void argus_model_free(argus_model_t * model);

/**
 * @brief Initializes an active execution session context on the model.
 * @param model Target loaded model.
 * @param params Context parameters (KV cache size, quantization, threads).
 * @return Opaque context session pointer, or NULL on failure.
 */
ARGUS_API argus_context_t * argus_context_init(argus_model_t * model, const argus_context_params_t * params);

/**
 * @brief Releases execution context structures.
 * @param ctx Target execution context.
 */
ARGUS_API void argus_context_free(argus_context_t * ctx);

/**
 * @brief Sets CPU thread counts for an existing context session.
 * Automatically updates main evaluation, speculative draft, and vocoder contexts if present.
 * @param ctx Target execution context.
 * @param n_threads Number of threads for single-token generation decoding.
 * @param n_threads_batch Number of threads for prompt and batch token processing.
 */
ARGUS_API void argus_set_n_threads(argus_context_t * ctx, int32_t n_threads, int32_t n_threads_batch);

/**
 * @brief Queries active thread count for single-token generation decoding.
 * @param ctx Target execution context.
 * @return Number of allocated generation threads, or -1 if context is invalid.
 */
ARGUS_API int32_t argus_get_n_threads(argus_context_t * ctx);

/**
 * @brief Queries active thread count for prompt and batch token processing.
 * @param ctx Target execution context.
 * @return Number of allocated batch threads, or -1 if context is invalid.
 */
ARGUS_API int32_t argus_get_n_threads_batch(argus_context_t * ctx);

/**
 * @brief Checks if speculative decoding draft context is initialized and active on this context session.
 * @param ctx Target execution context.
 * @return true if speculative draft context is active, false otherwise.
 */
ARGUS_API bool argus_context_has_draft(const argus_context_t * ctx);

// =========================================================================
// 3. Core Text Inference & Native Codec Speech Synthesis (TTS)
// =========================================================================

/**
 * @brief Parses and converts a raw character string into an array of token IDs.
 * This is a read-only concurrent operation on the model vocabulary.
 * @param model Reference model containing the vocabulary mapping.
 * @param text Absolute address pointing to the null-terminated UTF-8 input string.
 * @param out_tokens Destination buffer memory segment to receive integer token IDs.
 * @param max_tokens Allocation ceiling size limit of the provided output segment.
 * @param add_bos Prepend the model's explicit Beginning-Of-Sentence token metadata.
 * @return The absolute number of parsed tokens stored inside the buffer, or negative on failure.
 */
ARGUS_API int32_t argus_tokenize(const argus_model_t * model, const char * text, int32_t * out_tokens, int32_t max_tokens, bool add_bos);

/**
 * @brief Converts a standalone single token primitive back into raw text bytes.
 * This is a read-only concurrent operation on the model vocabulary.
 * @param model Reference model containing the vocabulary mapping.
 * @param token Raw numerical token identifier.
 * @param out_buf Destination character array buffer to populate.
 * @param buf_size Sizing constraint capacity limits of the provided target output buffer.
 * @return Written character byte length count inside the buffer segment.
 */
ARGUS_API int32_t argus_token_to_piece(const argus_model_t * model, int32_t token, char * out_buf, int32_t buf_size);

/**
 * @brief Retrieves the Beginning-Of-Sentence (BOS) token ID.
 * @param model Reference model containing the vocabulary mapping.
 * @return The BOS token ID, or -1 if not defined/failed.
 */
ARGUS_API int32_t argus_vocab_bos(const argus_model_t * model);

/**
 * @brief Retrieves the End-Of-Sentence (EOS) token ID.
 * @param model Reference model containing the vocabulary mapping.
 * @return The EOS token ID, or -1 if not defined/failed.
 */
ARGUS_API int32_t argus_vocab_eos(const argus_model_t * model);

/**
 * @brief Retrieves the End-Of-Turn (EOT) token ID.
 * @param model Reference model containing the vocabulary mapping.
 * @return The EOT token ID, or -1 if not defined/failed.
 */
ARGUS_API int32_t argus_vocab_eot(const argus_model_t * model);

/**
 * @brief Retrieves the Padding (PAD) token ID.
 * @param model Reference model containing the vocabulary mapping.
 * @return The PAD token ID, or -1 if not defined/failed.
 */
ARGUS_API int32_t argus_vocab_pad(const argus_model_t * model);

/**
 * @brief Retrieves the total size of the vocabulary.
 * @param model Reference model containing the vocabulary mapping.
 * @return The total token count, or -1 on failure.
 */
ARGUS_API int32_t argus_vocab_n_tokens(const argus_model_t * model);

/**
 * @brief Checks if a token is an End-Of-Generation (EOG) token.
 * @param model Reference model containing the vocabulary mapping.
 * @param token The token ID to inspect.
 * @return True if the token is an EOG token, false otherwise.
 */
ARGUS_API bool argus_vocab_is_eog(const argus_model_t * model, int32_t token);

/**
 * @brief Extracts model GGUF metadata string values by key name.
 * @param model Reference model containing the metadata.
 * @param key Null-terminated string key to look up.
 * @param buf Output character buffer to populate.
 * @param buf_size Character size capacity limits of the provided target output buffer.
 * @return String character length written, or negative on failure.
 */
ARGUS_API int32_t argus_model_meta_val_str(const argus_model_t * model, const char * key, char * buf, int32_t buf_size);

/**
 * @brief Retrieves the total count of metadata entries in the GGUF model.
 * @param model Reference model containing the metadata.
 * @return The total number of key-value pairs, or negative on failure.
 */
ARGUS_API int32_t argus_model_meta_count(const argus_model_t * model);

/**
 * @brief Retrieves the metadata key name at a specified index.
 * @param model Reference model containing the metadata.
 * @param index Registry index.
 * @param buf Output character buffer to populate.
 * @param buf_size Character size capacity limits of the provided target output buffer.
 * @return String character length written, or negative on failure.
 */
ARGUS_API int32_t argus_model_meta_key_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size);

/**
 * @brief Retrieves the metadata value string at a specified index.
 * @param model Reference model containing the metadata.
 * @param index Registry index.
 * @param buf Output character buffer to populate.
 * @param buf_size Character size capacity limits of the provided target output buffer.
 * @return String character length written, or negative on failure.
 */
ARGUS_API int32_t argus_model_meta_val_str_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size);

/**
 * @brief Retrieves the model's embedding dimension.
 * @param model Reference model weights handler.
 * @return The embedding dimension length, or -1 on failure/not defined.
 */
ARGUS_API int32_t argus_model_n_embd(const argus_model_t * model);

/**
 * @brief Retrieves the model's training context length limit.
 * @param model Reference model weights handler.
 * @return The training context size ceiling, or -1 on failure/not defined.
 */
ARGUS_API int32_t argus_model_n_ctx_train(const argus_model_t * model);

/**
 * @brief Retrieves the model's transformer layer count.
 * @param model Reference model weights handler.
 * @return The transformer layer count, or -1 on failure/not defined.
 */
ARGUS_API int32_t argus_model_n_layer(const argus_model_t * model);

/**
 * @brief Retrieves the model's attention query head count.
 * @param model Reference model weights handler.
 * @return The attention query head count, or -1 on failure/not defined.
 */
ARGUS_API int32_t argus_model_n_head(const argus_model_t * model);

/**
 * @brief Retrieves the model's attention key-value head count.
 * @param model Reference model weights handler.
 * @return The attention key-value head count, or -1 on failure/not defined.
 */
ARGUS_API int32_t argus_model_n_head_kv(const argus_model_t * model);

/**
 * @brief Retrieves the model's total parameter count.
 * @param model Reference model weights handler.
 * @return The total parameter count, or 0 on failure.
 */
ARGUS_API uint64_t argus_model_n_params(const argus_model_t * model);

/**
 * @brief Checks if the loaded GGUF model contains an encoder stack or non-causal topology.
 * @param model Reference model weights handler.
 * @return True if the model has an encoder architecture, false otherwise.
 */
ARGUS_API bool argus_model_has_encoder(const argus_model_t * model);

/**
 * @brief Retrieves the number of rotary position embedding dimensions per token (e.g. 4 for M-RoPE, 1 for standard 1D-RoPE).
 * @param model Reference model weights handler.
 * @return Number of position dimensions per token (e.g. 4 or 1), or -1 on failure.
 */
ARGUS_API int32_t argus_model_n_pos_per_embd(const argus_model_t * model);

/**
 * @brief Checks whether the loaded GGUF model uses Multimodal Rotary Position Embeddings (M-RoPE / IM-RoPE).
 * @param model Reference model weights handler.
 * @return True if model uses M-RoPE architecture, false otherwise.
 */
ARGUS_API bool argus_model_is_mrope(const argus_model_t * model);

/**
 * @brief Retrieves total loaded memory size of model weights in bytes.
 * @param model Reference model weights handler.
 * @return Total weight footprint in bytes, or 0 on failure.
 */
ARGUS_API uint64_t argus_model_size(const argus_model_t * model);

/**
 * @brief Retrieves human-readable model architecture description string.
 * @param model Reference model weights handler.
 * @param buf Output buffer array to populate.
 * @param buf_size Sizing constraint capacity limits of out buffer.
 * @return Character length written, or negative on failure.
 */
ARGUS_API int32_t argus_model_desc(const argus_model_t * model, char * buf, int32_t buf_size);

/**
 * @brief Calculates the KV cache memory cost in bytes per token for a loaded model.
 * Takes into account transformer layer count, KV head count, head dimension, and
 * KV cache quantization precision (type_k, type_v).
 * 
 * @param model Reference model handler.
 * @param type_k Key cache quantization type (argus_kv_type_t / ggml_type).
 * @param type_v Value cache quantization type (argus_kv_type_t / ggml_type).
 * @return Bytes required per token for the entire model KV cache stack, or -1 on error.
 */
ARGUS_API int64_t argus_model_kv_bytes_per_token(
    const argus_model_t * model, 
    int32_t type_k, 
    int32_t type_v
);

/**
 * @brief Calculates total estimated memory requirement (Model Weights + KV Cache) for a target context length.
 * 
 * @param model Reference model handler.
 * @param context_length Proposed context size in tokens (e.g. 65536).
 * @param type_k Key cache quantization type.
 * @param type_v Value cache quantization type.
 * @return Total estimated VRAM bytes required, or -1 on error.
 */
ARGUS_API int64_t argus_model_estimate_vram_bytes(
    const argus_model_t * model,
    int32_t context_length,
    int32_t type_k,
    int32_t type_v
);

/**
 * @brief Gets byte size for a block of elements of the specified GGML quantization type.
 * @param type GGML quantization type identifier (argus_kv_type_t / ggml_type).
 * @return Block size in bytes, or 0 for invalid type.
 */
ARGUS_API size_t argus_quant_type_size(int32_t type);

/**
 * @brief Gets element count per block for the specified GGML quantization type.
 * @param type GGML quantization type identifier (argus_kv_type_t / ggml_type).
 * @return Element count per block, or 0 for invalid type.
 */
ARGUS_API int32_t argus_quant_block_size(int32_t type);

/**
 * @brief Evaluates an unmanaged batch payload through the model's compute graph.
 * This is a synchronized mutating context operation.
 * @param ctx Reference execution context.
 * @param batch The specific structural collection of tokens, offsets, and sequence bindings.
 * @return Status code mapping (0 denotes clean execution, non-zero alerts structural errors).
 */
ARGUS_API int32_t argus_decode_batch(argus_context_t * ctx, const argus_token_batch_t * batch);

/**
 * @brief Retrieves the embeddings vector for a specific sequence ID.
 * This is a synchronized context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence tracking ID.
 * @param out_embeddings Flat floating-point buffer to receive the embedding vector.
 * @param max_floats Size of the output buffer in float elements.
 * @return Number of floats written to out_embeddings, or negative on failure.
 */
ARGUS_API int32_t argus_get_embeddings(argus_context_t * ctx, int32_t seq_id, float * out_embeddings, int32_t max_floats);

/**
 * @brief Samples a single token from the last computed layer logits matrix.
 * This is a synchronized mutating context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence track being sampled.
 * @param temperature Mathematical extraction entropy control value (<= 0.0f = greedy).
 * @param repeat_penalty Token frequency suppression multiplier factor (<= 1.0f = disabled).
 * @return The resolved token ID primitive, -1 on invalid args, or -2 if logits unavailable for seq_id.
 */
ARGUS_API int32_t argus_sample_token(argus_context_t * ctx, int32_t seq_id, float temperature, float repeat_penalty);

/**
 * @brief Samples a single token applying specified logit bias weightings.
 * This is a synchronized mutating context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence track being sampled.
 * @param temperature Mathematical extraction entropy control value (<= 0.0f = greedy).
 * @param repeat_penalty Token frequency suppression multiplier factor (<= 1.0f = disabled).
 * @param biases Contiguous unmanaged array containing target tokens and bias values.
 * @param bias_count Total count of biased tokens in the array.
 * @return The resolved token ID primitive, -1 on invalid args, or -2 if logits unavailable for seq_id.
 */
ARGUS_API int32_t argus_sample_token_with_bias(
    argus_context_t          * ctx, 
    int32_t                    seq_id, 
    float                      temperature, 
    float                      repeat_penalty, 
    const argus_logit_bias_t * biases, 
    int32_t                    bias_count
);

/**
 * @brief Samples a single token using the extended sampling hyperparameter envelope with zero-allocation caching.
 * This is a synchronized mutating context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence track being sampled.
 * @param sparams Extended sampler configuration parameters (top_p, min_p, top_k, temp, penalties, dry, seed).
 * @param biases Optional array of logit bias weightings.
 * @param bias_count Count of biased tokens.
 * @return The resolved token ID primitive, -1 on invalid args, or -2 if logits unavailable for seq_id.
 */
ARGUS_API int32_t argus_sample_token_ext(
    argus_context_t              * ctx,
    int32_t                        seq_id,
    const argus_sampler_params_t * sparams,
    const argus_logit_bias_t     * biases,
    int32_t                        bias_count
);

/**
 * @brief Resets the sampler state and clears penalty/DRY token history for a sequence slot.
 * This is a synchronized mutating context operation.
 * @param ctx Target execution context.
 * @param seq_id Target sequence slot (-1 for all sequences).
 * @return 0 on success, negative on failure.
 */
ARGUS_API int32_t argus_sampler_reset(argus_context_t * ctx, int32_t seq_id);

/**
 * @brief Primes a sequence slot's stateful sampler history with prompt or external tokens.
 * Accepted tokens are tracked for repetition penalties and DRY n-gram suppression.
 * This is a synchronized mutating context operation.
 * @param ctx Target execution context.
 * @param seq_id Target sequence slot.
 * @param tokens Array of token IDs to prime into history.
 * @param n_tokens Number of tokens in the array.
 * @return 0 on success, negative on failure.
 */
ARGUS_API int32_t argus_sampler_prime(argus_context_t * ctx, int32_t seq_id, const int32_t * tokens, int32_t n_tokens);

/**
 * @brief Truncates a sequence slot's sampler token history to the specified length.
 * Resets the sampler chain and replays the surviving prefix.
 * This is a synchronized mutating context operation.
 * @param ctx Target execution context.
 * @param seq_id Target sequence slot.
 * @param new_length Retained history token length.
 * @return 0 on success, negative on failure.
 */
ARGUS_API int32_t argus_sampler_truncate(argus_context_t * ctx, int32_t seq_id, int32_t new_length);

/**
 * @brief Prunes targeted token chains out of the active unmanaged L1 VRAM sequence cache.
 * Automatically synchronizes pruning across speculative draft contexts if present.
 * This is a synchronized mutating context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence track context slot to alter.
 * @param p0 Starting position offset cell parameter (-1 represents infinite tracking boundary bounds).
 * @param p1 Terminating position offset cell parameter.
 */
ARGUS_API void argus_kv_cache_clear_slot(argus_context_t * ctx, int32_t seq_id, int32_t p0, int32_t p1);

/**
 * @brief Queries the highest position offset currently allocated in the KV cache for a sequence slot.
 * This is a synchronized context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence slot tracking ID.
 * @return Largest position present in memory for sequence, or -1 if sequence slot is empty or invalid.
 */
ARGUS_API int32_t argus_kv_cache_seq_pos_max(const argus_context_t * ctx, int32_t seq_id);

/**
 * @brief Queries the lowest position offset currently allocated in the KV cache for a sequence slot.
 * This is a synchronized context operation.
 * @param ctx Reference execution context.
 * @param seq_id Target sequence slot tracking ID.
 * @return Smallest position present in memory for sequence, or -1 if sequence slot is empty or invalid.
 */
ARGUS_API int32_t argus_kv_cache_seq_pos_min(const argus_context_t * ctx, int32_t seq_id);

/**
 * @brief Evaluates and normalizes raw text input straight into an audio wave segment.
 * Generates speech natively via unified unmanaged Speech-LLM multi-codecs.
 * This is a synchronized context operation.
 * @param ctx Reference execution context containing loaded OuteTTS model states.
 * @param wavtokenizer_model Reference model containing the WavTokenizer vocoder.
 * @param text Null-terminated conversational stream string to read aloud.
 * @param voice_seed Integer layout seed determining vocal timbre and speaker baseline.
 * @param out_pcm Flat floating-point buffer array segment to hold 16kHz/24kHz normalized sample data.
 * @param max_samples Sizing ceiling constraints of the destination floating-point data buffer segment.
 * @return Absolute element sample count written directly to the memory address.
 */
ARGUS_API int32_t argus_synthesize_speech(argus_context_t * ctx, const argus_model_t * wavtokenizer_model, const char * text, int32_t voice_seed, float * out_pcm, int32_t max_samples, float * workspace, int64_t workspace_size_floats);

// =========================================================================
// 4. Audio Stream Transcription Subsystem (ASR / STT)
// =========================================================================

/**
 * @brief Allocates and initializes an independent Whisper acoustic evaluation framework.
 * @param params Parameters configuring model locations and processing boundaries.
 * @return Reference pointer to an active opaque instance handler, or NULL on failure.
 */
ARGUS_API argus_audio_context_t * argus_audio_init(const argus_audio_params_t * params);

/**
 * @brief Safely dismantles whisper memory graphs and releases system handlers.
 * @param ctx Reference pointer to the targeted audio engine context handler.
 */
ARGUS_API void argus_audio_free(argus_audio_context_t * ctx);

/**
 * @brief Sets CPU thread count for an existing Whisper audio context session.
 * @param ctx Reference pointer to the targeted audio engine context handler.
 * @param n_threads Number of threads for acoustic matrix calculations.
 */
ARGUS_API void argus_audio_set_n_threads(argus_audio_context_t * ctx, int32_t n_threads);

/**
 * @brief Queries active CPU thread count for Whisper audio context session.
 * @param ctx Reference pointer to the targeted audio engine context handler.
 * @return Number of allocated acoustic threads, or -1 if context is invalid.
 */
ARGUS_API int32_t argus_audio_get_n_threads(argus_audio_context_t * ctx);

/**
 * @brief Processes floating-point audio data arrays directly through Whisper networks.
 * This is a synchronized context operation.
 * @param ctx Reference whisper audio engine context handler.
 * @param pcm_data Pointer to contiguous array of normalized 32-bit floating-point samples (16kHz).
 * @param sample_count The total number of audio samples contained inside the array payload.
 * @param out_text Target text buffer to populate with transcribed conversational string segments.
 * @param max_chars Safety character constraint limit of the text array buffer.
 * @return Realized string length count written inside the buffer.
 */
ARGUS_API int32_t argus_transcribe_audio(argus_audio_context_t * ctx, const float * pcm_data, int32_t sample_count, char * out_text, int32_t max_chars);

// =========================================================================
// 5. Bleeding-Edge Multimodal LLM Operations (Phase 3)
// =========================================================================

/**
 * @brief Opaque handler representing an active multimodal projector session context.
 */
typedef struct argus_multimodal argus_multimodal_t;

/**
 * @brief Opaque handler representing raw/parsed media content in memory.
 */
typedef struct argus_bitmap argus_bitmap_t;

/**
 * @brief Opaque handler representing an active video decoder pipe.
 */
typedef struct argus_video argus_video_t;

/**
 * @brief Opaque handler representing a list of tokenized prompt chunks.
 */
typedef struct argus_input_chunks argus_input_chunks_t;

/**
 * @brief Initializes and loads a multimodal vision/audio/video projector.
 * @param model Reference to the pre-loaded base GGUF model.
 * @param params Projector configuration parameters.
 * @return Multimodal context pointer, or NULL on failure.
 */
ARGUS_API argus_multimodal_t * argus_multimodal_init(const argus_model_t * model, const argus_multimodal_params_t * params);

/**
 * @brief Releases multimodal projector resources.
 * @param mctx Target multimodal context pointer.
 */
ARGUS_API void argus_multimodal_free(argus_multimodal_t * mctx);

/**
 * @brief Checks if the loaded multimodal context supports vision/image inputs.
 */
ARGUS_API bool argus_multimodal_support_vision(const argus_multimodal_t * mctx);

/**
 * @brief Checks if the loaded multimodal context supports audio inputs.
 */
ARGUS_API bool argus_multimodal_support_audio(const argus_multimodal_t * mctx);

/**
 * @brief Checks if the loaded multimodal context supports video inputs.
 */
ARGUS_API bool argus_multimodal_support_video(const argus_multimodal_t * mctx);

/**
 * @brief Returns the expected audio sample rate of the projector (e.g. 16000).
 */
ARGUS_API int32_t argus_multimodal_get_audio_sample_rate(const argus_multimodal_t * mctx);

/**
 * @brief Creates a raw RGB image bitmap (RGBRGB... format).
 * @param width Image width.
 * @param height Image height.
 * @param rgb_data Contiguous array of raw 24-bit RGB values.
 * @return Bitmap pointer, or NULL on failure.
 */
ARGUS_API argus_bitmap_t * argus_bitmap_from_rgb(uint32_t width, uint32_t height, const uint8_t * rgb_data);

/**
 * @brief Creates a raw float PCM audio sample bitmap.
 * @param pcm_data Contiguous array of 32-bit floating point audio samples.
 * @param n_samples Total count of audio samples.
 * @return Bitmap pointer, or NULL on failure.
 */
ARGUS_API argus_bitmap_t * argus_bitmap_from_pcm(const float * pcm_data, int32_t n_samples);

/**
 * @brief Automatically loads and processes media (image/audio) from a local file.
 * @param mctx Active multimodal projector context.
 * @param path Local path to the media file.
 * @param placeholder If true, creates a placeholder bitmap for counting tokens without decoding.
 * @return Bitmap pointer, or NULL on failure.
 */
ARGUS_API argus_bitmap_t * argus_bitmap_load_file(argus_multimodal_t * mctx, const char * path, bool placeholder);

/**
 * @brief Automatically loads and processes media (image/audio) from an in-memory buffer.
 * @param mctx Active multimodal projector context.
 * @param buffer Contiguous byte array holding the media file.
 * @param size Total count of bytes in the buffer.
 * @param placeholder If true, creates a placeholder bitmap for counting tokens without decoding.
 * @return Bitmap pointer, or NULL on failure.
 */
ARGUS_API argus_bitmap_t * argus_bitmap_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, bool placeholder);

/**
 * @brief Releases unmanaged bitmap memory resources.
 * @param bitmap Target bitmap pointer.
 */
ARGUS_API void argus_bitmap_free(argus_bitmap_t * bitmap);

/**
 * @brief Prepares a video processing session from a file.
 * @param mctx Active multimodal projector context.
 * @param path Local path to the video file.
 * @param fps_target Target frame extraction rate (<= 0 for native video FPS).
 * @param timestamp_interval_ms Interval for inserting timestamp text chunks (<= 0 to disable).
 * @return Video context pointer, or NULL on failure.
 */
ARGUS_API argus_video_t * argus_video_load_file(argus_multimodal_t * mctx, const char * path, float fps_target, int64_t timestamp_interval_ms);

/**
 * @brief Prepares a video processing session from an in-memory buffer.
 * @param mctx Active multimodal projector context.
 * @param buffer Contiguous byte array holding the video file contents.
 * @param size Total count of bytes in the buffer.
 * @param fps_target Target frame extraction rate (<= 0 for native video FPS).
 * @param timestamp_interval_ms Interval for inserting timestamp text chunks (<= 0 to disable).
 * @return Video context pointer, or NULL on failure.
 */
ARGUS_API argus_video_t * argus_video_load_buffer(argus_multimodal_t * mctx, const uint8_t * buffer, int32_t size, float fps_target, int64_t timestamp_interval_ms);

/**
 * @brief Releases video decoder context.
 * @param video Target video pointer.
 */
ARGUS_API void argus_video_free(argus_video_t * video);

/**
 * @brief Iterates the video stream to extract the next bitmap frame or text timestamp chunk.
 * Exactly one of out_bitmap or out_text will be set per successful invocation.
 * @param video Active video pointer.
 * @param out_bitmap Destination address to receive the newly decoded frame bitmap pointer.
 * @param out_text Destination address to write any timestamp text strings.
 * @param max_chars Safety character constraint limit of the out_text buffer.
 * @return 0 on success, -1 on EOF, -2 on decoding error.
 */
ARGUS_API int32_t argus_video_read_next(argus_video_t * video, argus_bitmap_t ** out_bitmap, char * out_text, int32_t max_chars);

/**
 * @brief Allocates an empty input chunks container list.
 * @return Input chunks container pointer.
 */
ARGUS_API argus_input_chunks_t * argus_input_chunks_init(void);

/**
 * @brief Frees the input chunks container and all populated chunks inside it.
 * @param chunks Target chunks container pointer.
 */
ARGUS_API void argus_input_chunks_free(argus_input_chunks_t * chunks);

/**
 * @brief Tokenizes the text prompt and media bitmaps into unified sequential chunks.
 * Replaces prompt markers (default: "<__media__>") with media embeddings/token grids.
 * @param mctx Active multimodal projector context.
 * @param output Destination chunks container pointer.
 * @param text The input prompt text.
 * @param add_bos Prepend the model's Beginning-Of-Sentence token.
 * @param bitmaps Array of media bitmaps to replace markers in order.
 * @param n_bitmaps Count of bitmaps in the array.
 * @return 0 on success, non-zero on tokenization/preprocessing failure.
 */
ARGUS_API int32_t argus_multimodal_tokenize(
    argus_multimodal_t * mctx,
    argus_input_chunks_t * output,
    const char * text,
    bool add_bos,
    const argus_bitmap_t ** bitmaps,
    int32_t n_bitmaps);

/**
 * @brief Evaluates the tokenized chunks through both multimodal projectors and LLM decoders.
 * Runs GPU batch encoding for media and schedules correct M-RoPE/Attention attention grids.
 * @param mctx Active multimodal projector context.
 * @param ctx Active text execution context.
 * @param chunks Populated input chunks container pointer.
 * @param n_past Current KV cache offset position.
 * @param seq_id Target sequence tracking ID.
 * @param n_batch Batch evaluation size constraint.
 * @param logits_last Request logits calculation solely on the terminal token.
 * @param out_new_n_past Destination address to write the updated KV cache offset.
 * @return 0 on success, non-zero on evaluation failure.
 */
ARGUS_API int32_t argus_eval_multimodal_chunks(
    argus_multimodal_t * mctx,
    argus_context_t * ctx,
    const argus_input_chunks_t * chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t * out_new_n_past);

#ifdef __cplusplus
}
#endif

#endif // LIBARGUS_H
