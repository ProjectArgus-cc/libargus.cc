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
import java.io.InputStream;

/**
 * Low-level MethodHandles linked to the compiled libargus shared library.
 * Implements dynamic library loading and bindings mapping.
 */
public final class ArgusBindings {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    static {
        String customPath = System.getProperty("cc.projectargus.libargus.path");
        if (customPath != null) {
            System.load(Paths.get(customPath).toAbsolutePath().toString());
        } else {
            if (!tryLoadFromResources()) {
                try {
                    System.loadLibrary("argus");
                } catch (UnsatisfiedLinkError e) {
                    // Fallback: try finding the shared library in common local build spots
                    String userDir = System.getProperty("user.dir");
                    String libName = System.mapLibraryName("argus");
                    File localLib = Paths.get(userDir, "build", libName).toFile();
                    if (localLib.exists()) {
                        System.load(localLib.getAbsolutePath());
                    } else {
                        localLib = Paths.get(userDir, "..", "build", libName).toFile(); // check workspace parent build
                        if (localLib.exists()) {
                            System.load(localLib.getAbsolutePath());
                        } else {
                            throw new UnsatisfiedLinkError("Could not locate native " + libName + 
                                " in java.library.path, classpath resources, or workspace build output. Please specify -Dcc.projectargus.libargus.path");
                        }
                    }
                }
            }
        }
        LOOKUP = SymbolLookup.loaderLookup();
    }

    private static boolean tryLoadFromResources() {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();
        
        String osDir;
        if (osName.contains("linux")) {
            osDir = "linux-" + osArch;
        } else if (osName.contains("windows")) {
            osDir = "windows-" + osArch;
        } else if (osName.contains("mac")) {
            osDir = "macos-" + osArch;
        } else {
            return false;
        }
        
        String libName = System.mapLibraryName("argus");
        String resourcePath = "/natives/" + osDir + "/" + libName;
        
        try (InputStream is = ArgusBindings.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                return false;
            }
            
            // Create a temporary file that will be deleted on exit
            File tempFile = File.createTempFile("libargus_", "_" + libName);
            tempFile.deleteOnExit();
            
            // Copy contents
            Files.copy(is, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            
            // Load the unpacked library
            System.load(tempFile.getAbsolutePath());
            return true;
        } catch (Exception e) {
            System.err.println("Warning: Failed to extract and load " + libName + " from classpath resources: " + e.getMessage());
            return false;
        }
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
        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN)
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

    // Tokenization
    public static final MethodHandle argus_tokenize = bind("argus_tokenize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, 
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN)
    );

    public static final MethodHandle argus_token_to_piece = bind("argus_token_to_piece",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );

    // Evaluation & Pruning
    public static final MethodHandle argus_decode_batch = bind("argus_decode_batch",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );

    public static final MethodHandle argus_sample_token = bind("argus_sample_token",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, 
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT)
    );

    public static final MethodHandle argus_kv_cache_clear_slot = bind("argus_kv_cache_clear_slot",
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
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
