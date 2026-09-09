package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Handles the process-global compute backends (CUDA, CPU, Metal) registry and telemetry.
 */
public final class ArgusBackend {
    public static final long FEATURE_CPU_ACCEL = 1L << 0;
    public static final long FEATURE_CPU       = FEATURE_CPU_ACCEL;
    public static final long FEATURE_CUDA      = 1L << 1;
    public static final long FEATURE_ROCM      = 1L << 2;
    public static final long FEATURE_VULKAN    = 1L << 3;
    public static final long FEATURE_METAL     = 1L << 4;
    public static final long FEATURE_SYCL      = 1L << 5;

    private static boolean initialized = false;

    private ArgusBackend() {}

    public static synchronized boolean init() {
        return init(ArgusBindings.EXTRACTED_DIR);
    }

    /**
     * Initializes the process-global hardware execution registry with a custom backend plugin path.
     * Must be called once before loading models or executing transcription/speech.
     *
     * @param customPluginPath directory path to search for dynamic ggml plugin libraries (like CUDA)
     * @return true if backend registers successfully, false otherwise.
     */
    public static synchronized boolean init(String customPluginPath) {
        if (initialized) {
            return true;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment pathSeg = (customPluginPath != null && !customPluginPath.isEmpty())
                ? localArena.allocateFrom(customPluginPath)
                : MemorySegment.NULL;
            initialized = (boolean) ArgusBindings.argus_backend_init.invokeExact(pathSeg);
            return initialized;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_init", t);
        }
    }

    /**
     * Queries whether the process-global native compute backend registry is currently initialized.
     *
     * @return true if backend is initialized and active, false otherwise
     */
    public static boolean isInitialized() {
        try {
            return (boolean) ArgusBindings.argus_backend_is_initialized.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_is_initialized", t);
        }
    }

    /**
     * Queries the compile-time feature bitmask representing backends and acceleration built into the native binary.
     *
     * @return bitmask combining FEATURE_* flags
     */
    public static long getBuildFeatures() {
        try {
            return (long) ArgusBindings.argus_build_features.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_build_features", t);
        }
    }

    /**
     * Checks if a specific feature flag is present in the native library build.
     *
     * @param featureMask feature flag bitmask (e.g. FEATURE_CPU_ACCEL, FEATURE_CUDA)
     * @return true if enabled at compile time
     */
    public static boolean hasFeature(long featureMask) {
        return (getBuildFeatures() & featureMask) != 0;
    }

    /**
     * Deallocates global backend context registration and frees active compute pools.
     */
    public static synchronized void free() {
        if (!initialized) {
            return;
        }
        try {
            ArgusBindings.argus_backend_free.invokeExact();
            initialized = false;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_free", t);
        }
    }

    /**
     * @return count of available hardware backends.
     */
    public static int getCount() {
        try {
            return (int) ArgusBindings.argus_backend_get_count.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_get_count", t);
        }
    }

    /**
     * Query driver registry name at index.
     * @param index backend index.
     * @return dynamic backend name (e.g. "CUDA0", "CPU").
     */
    public static String getName(int index) {
        try {
            MemorySegment namePtr = (MemorySegment) ArgusBindings.argus_backend_get_name.invokeExact(index);
            if (namePtr.equals(MemorySegment.NULL)) {
                return "UNKNOWN";
            }
            return namePtr.reinterpret(256).getString(0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_get_name", t);
        }
    }

    /**
     * Utility listing all available backends on the machine.
     * @return list of backend driver names.
     */
    public static List<String> getAvailableBackends() {
        int count = getCount();
        if (count <= 0) {
            return Collections.emptyList();
        }
        List<String> list = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            list.add(getName(i));
        }
        return Collections.unmodifiableList(list);
    }

    /**
     * Returns the compiled framework version string.
     */
    public static String getVersion() {
        return ArgusBindings.VERSION;
    }

    /**
     * Isolated CLI entry point verifying SPI native library extraction and build feature masks.
     *
     * @param args optional args[0] specifies expected backend target: "cpu", "cuda", "rocm", "vulkan", "metal"
     */
    public static void main(String[] args) {
        String expectedTarget = args.length > 0 ? args[0] : null;
        System.out.println("[ArgusBackend] Version: " + getVersion());
        System.out.println("[ArgusBackend] Extracted/Loaded dir: " + ArgusBindings.EXTRACTED_DIR);
        long features = getBuildFeatures();
        System.out.println("[ArgusBackend] Build features bitmask: 0x" + Long.toHexString(features));
        if ((features & FEATURE_CPU) != 0) System.out.println("  - CPU");
        if ((features & FEATURE_CUDA) != 0) System.out.println("  - CUDA");
        if ((features & FEATURE_ROCM) != 0) System.out.println("  - ROCm/HIP");
        if ((features & FEATURE_VULKAN) != 0) System.out.println("  - Vulkan");
        if ((features & FEATURE_METAL) != 0) System.out.println("  - Metal");

        if (expectedTarget != null && !expectedTarget.isEmpty()) {
            long expected = expectedFeatureMask(expectedTarget);
            if (features != expected) {
                System.err.println("ARGUS_FEATURE_MISMATCH expected=" + expected + " actual=" + features);
                System.exit(3);
            }
        }
    }

    public static long expectedFeatureMask(String target) {
        return switch (target.toLowerCase(java.util.Locale.ROOT)) {
            case "cpu" -> FEATURE_CPU;
            case "cuda" -> FEATURE_CPU | FEATURE_CUDA;
            case "rocm", "hip" -> FEATURE_CPU | FEATURE_ROCM;
            case "vulkan" -> FEATURE_CPU | FEATURE_VULKAN;
            case "metal" -> FEATURE_CPU | FEATURE_METAL;
            default -> throw new IllegalArgumentException("Unknown backend: " + target);
        };
    }

    /** Build metadata only; does not initialize drivers or perform device execution. */
    public static java.util.Map<String, String> getBuildInfo() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(4096);
            int required = (int) ArgusBindings.argus_build_info_copy.invokeExact(out, 4096);
            if (required < 0 || required >= 4096) throw new IllegalStateException("Invalid build metadata size");
            java.util.Map<String, String> info = new java.util.LinkedHashMap<>();
            for (String line : out.getString(0).split("\\n")) {
                int separator = line.indexOf('=');
                if (separator <= 0 || info.put(line.substring(0, separator), line.substring(separator + 1)) != null)
                    throw new IllegalStateException("Invalid build metadata");
            }
            info.put("features", Long.toString(getBuildFeatures()));
            if (!cc.projectargus.libargus.internal.ArgusLayouts.diagnosticAbi().equals(info.get("abi_layouts")))
                throw new IllegalStateException("Native ABI layout differs from Java layouts");
            return java.util.Map.copyOf(info);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new IllegalStateException("Cannot read native build metadata", t);
        }
    }
}
