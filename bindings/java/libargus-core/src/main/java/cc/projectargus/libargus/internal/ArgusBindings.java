package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusNativeException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Low-level MethodHandles linked to the compiled libargus shared library.
 * Delegates native library discovery and SPI extraction to {@link ArgusNativeLoader}.
 */
public final class ArgusBindings {
    private static final Linker LINKER = Linker.nativeLinker();
    public static final String VERSION;
    private static final SymbolLookup LOOKUP;
    public static final String EXTRACTED_DIR;

    public static String loadedLibraryPath() {
        return ArgusNativeLoader.loadedLibraryPath();
    }

    public static String loadedProvider() {
        return ArgusNativeLoader.loadedProvider();
    }

    static {
        ArgusNativeLoader.ensureLoaded();
        VERSION = ArgusNativeLoader.VERSION;
        EXTRACTED_DIR = ArgusNativeLoader.EXTRACTED_DIR;
        SymbolLookup loaderLookup = SymbolLookup.loaderLookup();
        String libPath = ArgusNativeLoader.loadedLibraryPath();
        if (libPath != null && !libPath.startsWith("system-path:")) {
            SymbolLookup fileLookup = SymbolLookup.libraryLookup(java.nio.file.Path.of(libPath), Arena.global());
            LOOKUP = name -> loaderLookup.find(name).or(() -> fileLookup.find(name));
        } else {
            LOOKUP = loaderLookup;
        }
    }

    private ArgusBindings() {}

    private static MethodHandle bind(String name, FunctionDescriptor desc, Linker.Option... options) {
        return LINKER.downcallHandle(
            LOOKUP.find(name)
                  .orElseThrow(() -> new NoSuchMethodError("Failed to resolve native symbol: " + name)),
            desc,
            options
        );
    }

    public static final java.util.Set<String> CRITICAL_ALLOWLIST = java.util.Collections.unmodifiableSet(
        new java.util.HashSet<>(java.util.Arrays.asList(
            "argus_build_features",
            "argus_abort_flag_is_requested",
            "argus_last_error_code",
            "argus_clear_error"
        ))
    );

    static MethodHandle bindCritical(String name, FunctionDescriptor desc) {
        if (!CRITICAL_ALLOWLIST.contains(name)) {
            throw new SecurityException("Symbol '" + name + "' is not permitted in Panama critical allowlist!");
        }
        return bind(name, desc, Linker.Option.critical(false));
    }

    // Lifecycle
    public static final MethodHandle argus_backend_init = bind("argus_backend_init",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_version = bind("argus_version",
        FunctionDescriptor.of(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_backend_free = bind("argus_backend_free",
        FunctionDescriptor.ofVoid()
    );

    public static final MethodHandle argus_backend_get_count = bind("argus_backend_get_count",
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_backend_get_name = bind("argus_backend_get_name",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    // Logging & Diagnostics
    public static final MethodHandle argus_set_log_level = bind("argus_set_log_level",
        FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_get_log_level = bind("argus_get_log_level",
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_set_log_callback = bind("argus_set_log_callback",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_log_in_flight_count = bind("argus_log_in_flight_count",
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );

    // Text Models
    public static final MethodHandle argus_model_load = bind("argus_model_load",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_load_fd = bind("argus_model_load_fd",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_free = bind("argus_model_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    // Contexts
    public static final MethodHandle argus_context_init = bind("argus_context_init",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_free = bind("argus_context_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_get_model = bind("argus_context_get_model",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_set_n_threads = bind("argus_set_n_threads",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_get_n_threads = bind("argus_get_n_threads",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_get_n_threads_batch = bind("argus_get_n_threads_batch",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_has_draft = bind("argus_context_has_draft",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_get_seq_max = bind("argus_context_get_seq_max",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_get_perf = bind("argus_context_get_perf",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_reset_perf = bind("argus_context_reset_perf",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_context_set_causal_attn = bind("argus_context_set_causal_attn",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_context_get_causal_attn = bind("argus_context_get_causal_attn",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    // Tokenization
    public static final MethodHandle argus_tokenize = bind("argus_tokenize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, 
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_token_to_piece = bind("argus_token_to_piece",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_vocab_bos = bind("argus_vocab_bos",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_vocab_eos = bind("argus_vocab_eos",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_vocab_eot = bind("argus_vocab_eot",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_vocab_pad = bind("argus_vocab_pad",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_vocab_n_tokens = bind("argus_vocab_n_tokens",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_vocab_is_eog = bind("argus_vocab_is_eog",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_meta_val_str = bind("argus_model_meta_val_str",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, 
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_meta_val_str_n = bind("argus_model_meta_val_str_n",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_meta_count = bind("argus_model_meta_count",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_meta_key_by_index = bind("argus_model_meta_key_by_index",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_meta_val_str_by_index = bind("argus_model_meta_val_str_by_index",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_n_embd = bind("argus_model_n_embd",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_embd_out = bind("argus_model_n_embd_out",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_embd_inp = bind("argus_model_n_embd_inp",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_ctx_train = bind("argus_model_n_ctx_train",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_layer = bind("argus_model_n_layer",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_head = bind("argus_model_n_head",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_head_kv = bind("argus_model_n_head_kv",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_params = bind("argus_model_n_params",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_has_encoder = bind("argus_model_has_encoder",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_n_pos_per_embd = bind("argus_model_n_pos_per_embd",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_is_mrope = bind("argus_model_is_mrope",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_is_recurrent = bind("argus_model_is_recurrent",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_is_hybrid = bind("argus_model_is_hybrid",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_is_diffusion = bind("argus_model_is_diffusion",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_size = bind("argus_model_size",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_desc = bind("argus_model_desc",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_kv_bytes_per_token = bind("argus_model_kv_bytes_per_token",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_model_estimate_vram_bytes = bind("argus_model_estimate_vram_bytes",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_quant_type_size = bind("argus_quant_type_size",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_quant_block_size = bind("argus_quant_block_size",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_build_info_copy = bind("argus_build_info_copy",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    public static final MethodHandle argus_decode_tokens = bind("argus_decode_tokens",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS));

    // Evaluation & Pruning
    public static final MethodHandle argus_decode_batch = bind("argus_decode_batch",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_sample_token = bind("argus_sample_token",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, 
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT)
    );

    public static final MethodHandle argus_sample_token_with_bias = bind("argus_sample_token_with_bias",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sample_token_ext = bind("argus_sample_token_ext",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_reset = bind("argus_sampler_reset",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_prime = bind("argus_sampler_prime",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_truncate = bind("argus_sampler_truncate",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_get_history_count = bind("argus_sampler_get_history_count",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_has_pending = bind("argus_sampler_has_pending",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_sampler_discard_pending = bind("argus_sampler_discard_pending",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_kv_cache_clear_slot = bind("argus_kv_cache_clear_slot",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_kv_cache_can_shift = bind("argus_kv_cache_can_shift",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_kv_cache_seq_add = bind("argus_kv_cache_seq_add",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_kv_cache_seq_cp = bind("argus_kv_cache_seq_cp",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_kv_cache_seq_pos_max = bind("argus_kv_cache_seq_pos_max",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_kv_cache_seq_pos_min = bind("argus_kv_cache_seq_pos_min",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_get_embeddings = bind("argus_get_embeddings",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_eval_embeddings = bind("argus_eval_embeddings",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    // Speech Synthesis
    public static final MethodHandle argus_synthesize_speech = bind("argus_synthesize_speech",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
    );

    public static final MethodHandle argus_synthesize_speech_n = bind("argus_synthesize_speech_n",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
    );

    // Whisper Audio (STT)
    public static final MethodHandle argus_audio_init = bind("argus_audio_init",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_audio_free = bind("argus_audio_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_audio_set_n_threads = bind("argus_audio_set_n_threads",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_audio_get_n_threads = bind("argus_audio_get_n_threads",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_transcribe_audio = bind("argus_transcribe_audio",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    // Multimodal LLM Operations (Phase 3)
    public static final MethodHandle argus_multimodal_init = bind("argus_multimodal_init",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_free = bind("argus_multimodal_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_retain = bind("argus_multimodal_retain",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_release = bind("argus_multimodal_release",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_support_vision = bind("argus_multimodal_support_vision",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_support_audio = bind("argus_multimodal_support_audio",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_support_video = bind("argus_multimodal_support_video",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_get_audio_sample_rate = bind("argus_multimodal_get_audio_sample_rate",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_bitmap_from_rgb = bind("argus_bitmap_from_rgb",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_bitmap_from_pcm = bind("argus_bitmap_from_pcm",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_bitmap_load_file = bind("argus_bitmap_load_file",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_bitmap_load_buffer = bind("argus_bitmap_load_buffer",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_bitmap_free = bind("argus_bitmap_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_bitmap_set_mergeable = bind("argus_bitmap_set_mergeable",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_video_load_file = bind("argus_video_load_file",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG)
    );

    public static final MethodHandle argus_video_load_buffer = bind("argus_video_load_buffer",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG)
    );

    public static final MethodHandle argus_video_free = bind("argus_video_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_video_read_next = bind("argus_video_read_next",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_input_chunks_init = bind("argus_input_chunks_init",
        FunctionDescriptor.of(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_input_chunks_free = bind("argus_input_chunks_free",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_multimodal_tokenize = bind("argus_multimodal_tokenize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_eval_multimodal_chunks = bind("argus_eval_multimodal_chunks",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    // Error Handling & Diagnostic Status
    public static final MethodHandle argus_last_error_code = bindCritical("argus_last_error_code",
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_last_error_message = bind("argus_last_error_message",
        FunctionDescriptor.of(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_last_error_message_copy = bind("argus_last_error_message_copy",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    public static final MethodHandle argus_clear_error = bindCritical("argus_clear_error",
        FunctionDescriptor.ofVoid()
    );

    public static final MethodHandle argus_build_features = bindCritical("argus_build_features",
        FunctionDescriptor.of(ValueLayout.JAVA_LONG)
    );

    public static final MethodHandle argus_backend_is_initialized = bind("argus_backend_is_initialized",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN)
    );

    // Model Retain & Release
    public static final MethodHandle argus_model_retain = bind("argus_model_retain",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_model_release = bind("argus_model_release",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    // Abort Flag
    public static final MethodHandle argus_abort_flag_create = bind("argus_abort_flag_create",
        FunctionDescriptor.of(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_abort_flag_request = bind("argus_abort_flag_request",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_abort_flag_reset = bind("argus_abort_flag_reset",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_abort_flag_is_requested = bindCritical("argus_abort_flag_is_requested",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_abort_flag_retain = bind("argus_abort_flag_retain",
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_abort_flag_release = bind("argus_abort_flag_release",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    // Length-bearing Tokenization
    public static final MethodHandle argus_tokenize_n = bind("argus_tokenize_n",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_multimodal_tokenize_n = bind("argus_multimodal_tokenize_n",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    static {
        ArgusNativeException.registerErrorResolver(new ArgusNativeException.ErrorResolver() {
            @Override
            public int lastErrorCode() {
                try {
                    return (int) argus_last_error_code.invokeExact();
                } catch (Throwable t) {
                    return ArgusNativeException.ERROR_INTERNAL;
                }
            }

            @Override
            public String lastErrorMessage() {
                try (Arena localArena = Arena.ofConfined()) {
                    MemorySegment buf = localArena.allocate(512);
                    int len = (int) argus_last_error_message_copy.invokeExact(buf, 512);
                    if (len > 0) {
                        return buf.getString(0);
                    }
                } catch (Throwable ignored) {
                }
                return "Unknown native failure";
            }

            @Override
            public void clearError() {
                try {
                    argus_clear_error.invokeExact();
                } catch (Throwable ignored) {
                }
            }
        });
    }
}
