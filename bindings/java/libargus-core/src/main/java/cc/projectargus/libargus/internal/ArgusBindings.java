package cc.projectargus.libargus.internal;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.util.ServiceLoader;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;


/**
 * Low-level MethodHandles linked to the compiled libargus shared library.
 * Implements dynamic library loading and bindings mapping.
 */
public final class ArgusBindings {
    private static final Linker LINKER = Linker.nativeLinker();
    public static final String VERSION;
    private static final SymbolLookup LOOKUP;
    public static final String EXTRACTED_DIR;

    static {
        // 1. Resolve version from classpath resource
        String resolvedVersion = "0.0.0";
        try (InputStream is = ArgusBindings.class.getResourceAsStream("/version.txt")) {
            if (is != null) {
                resolvedVersion = new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).trim();
            }
        } catch (IOException e) {
            // ignore
        }
        VERSION = resolvedVersion;

        // 2. Load native library
        String resolvedExtractedDir = null;
        String customPath = System.getProperty("cc.projectargus.libargus.path");
        if (customPath != null) {
            java.nio.file.Path path = Paths.get(customPath).toAbsolutePath();
            System.load(path.toString());
            resolvedExtractedDir = path.getParent().toString();
        } else {
            String tempExtracted = tryLoadFromSPI();
            if (tempExtracted != null) {
                resolvedExtractedDir = tempExtracted;
            } else {
                // 3. Try standard java.library.path / OS runtime library loader
                boolean loaded = false;
                try {
                    System.loadLibrary("argus");
                    loaded = true;
                } catch (UnsatisfiedLinkError e) {
                    // Fall back to bounded local workspace build directory search
                }

                if (!loaded) {
                    String userDir = System.getProperty("user.dir");
                    String libName = System.mapLibraryName("argus");
                    File currentDir = new File(userDir).getAbsoluteFile();
                    int maxDepth = 3;
                    while (currentDir != null && maxDepth-- >= 0) {
                        File buildDir = new File(currentDir, "build");
                        File localLibInLib = new File(new File(buildDir, "lib"), libName);
                        File localLibInBin = new File(new File(buildDir, "bin"), libName);
                        File localLib = new File(buildDir, libName);

                        File targetLib = null;
                        if (localLibInLib.exists()) {
                            targetLib = localLibInLib;
                        } else if (localLibInBin.exists()) {
                            targetLib = localLibInBin;
                        } else if (localLib.exists()) {
                            targetLib = localLib;
                        }

                        if (targetLib != null) {
                            System.load(targetLib.getAbsolutePath());
                            resolvedExtractedDir = targetLib.getParent();
                            loaded = true;
                            break;
                        }
                        currentDir = currentDir.getParentFile();
                    }
                }

                if (!loaded) {
                    String libName = System.mapLibraryName("argus");
                    throw new UnsatisfiedLinkError("Could not locate native " + libName + 
                        " via -Dcc.projectargus.libargus.path, SPI classpath providers, java.library.path (System.loadLibrary), or local workspace build/ directory. Please ensure native libraries are on java.library.path or specify -Dcc.projectargus.libargus.path");
                }
            }
        }
        EXTRACTED_DIR = resolvedExtractedDir;
        LOOKUP = SymbolLookup.loaderLookup();
    }

    private static String tryLoadFromSPI() {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();

        String normalizedOs;
        if (osName.contains("linux")) {
            normalizedOs = "linux";
        } else if (osName.contains("windows")) {
            normalizedOs = "windows";
        } else if (osName.contains("mac")) {
            normalizedOs = "macos";
        } else {
            normalizedOs = osName;
        }

        String normalizedArch;
        if (osArch.equals("amd64") || osArch.equals("x86_64")) {
            normalizedArch = "amd64";
        } else if (osArch.equals("aarch64") || osArch.equals("arm64")) {
            normalizedArch = "aarch64";
        } else {
            normalizedArch = osArch;
        }

        ServiceLoader<NativeLibraryProvider> loader = ServiceLoader.load(NativeLibraryProvider.class);
        List<NativeLibraryProvider> providers = new ArrayList<>();
        for (NativeLibraryProvider provider : loader) {
            String pOs = provider.getOs().toLowerCase();
            String pArch = provider.getArch().toLowerCase();
            
            boolean osMatch = pOs.contains(normalizedOs) || normalizedOs.contains(pOs);
            boolean archMatch = false;
            if ((normalizedArch.equals("amd64") && (pArch.equals("amd64") || pArch.equals("x86_64") || pArch.equals("x64"))) ||
                (normalizedArch.equals("aarch64") && (pArch.equals("aarch64") || pArch.equals("arm64"))) ||
                pArch.equals(normalizedArch)) {
                archMatch = true;
            }

            if (osMatch && archMatch) {
                providers.add(provider);
            }
        }

        if (providers.isEmpty()) {
            return null;
        }

        // Sort by priority descending
        providers.sort(new Comparator<NativeLibraryProvider>() {
            @Override
            public int compare(NativeLibraryProvider p1, NativeLibraryProvider p2) {
                return Integer.compare(p2.getPriority(), p1.getPriority());
            }
        });

        // Resolve output extraction cache directory
        String customDir = System.getProperty("cc.projectargus.libargus.nativeDir");
        File destDir;
        if (customDir != null && !customDir.trim().isEmpty()) {
            destDir = new File(customDir.trim());
        } else {
            destDir = new File(System.getProperty("java.io.tmpdir"), "argus_native_cache");
        }

        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new RuntimeException("Failed to create native extraction directory: " + destDir);
        }

        for (NativeLibraryProvider provider : providers) {
            String libVersionName = System.mapLibraryName("argus-" + provider.getBackend() + "-" + VERSION);
            File targetFile = new File(destDir, libVersionName);

            try {
                if (targetFile.exists() && targetFile.length() > 0) {
                    System.load(targetFile.getAbsolutePath());
                    return destDir.getAbsolutePath();
                }

                File tempFile = File.createTempFile("libargus_extract_", ".tmp", destDir);
                try {
                    try (InputStream is = provider.getLibraryStream()) {
                        if (is == null) {
                            continue;
                        }
                        Files.copy(is, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    }
                    Files.move(tempFile.toPath(), targetFile.toPath(),
                        StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
                } catch (IOException e) {
                    if (!targetFile.exists() || targetFile.length() == 0) {
                        throw e;
                    }
                } finally {
                    if (tempFile.exists()) {
                        tempFile.delete();
                    }
                }

                System.load(targetFile.getAbsolutePath());
                return destDir.getAbsolutePath();
            } catch (UnsatisfiedLinkError | Exception e) {
                System.err.println("Warning: Failed to load native library from provider [" + 
                    provider.getClass().getName() + " | Backend: " + provider.getBackend() + "]: " + e.getMessage());
            }
        }

        return null;
    }


    private ArgusBindings() {}

    private static MethodHandle bind(String name, FunctionDescriptor desc) {
        return LINKER.downcallHandle(
            LOOKUP.find(name)
                  .orElseThrow(() -> new NoSuchMethodError("Failed to resolve native symbol: " + name)),
            desc
        );
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

    // Text Models
    public static final MethodHandle argus_model_load = bind("argus_model_load",
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
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
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
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

    // Speech Synthesis
    public static final MethodHandle argus_synthesize_speech = bind("argus_synthesize_speech",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
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
}
