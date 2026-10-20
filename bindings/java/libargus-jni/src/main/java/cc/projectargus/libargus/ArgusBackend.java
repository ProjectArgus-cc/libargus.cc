package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Handles the process-global compute backends (CUDA, CPU, Metal, ROCm, Vulkan) registry and telemetry
 * via the JDK 17+ / Android-compatible JNI bridge.
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
        return init(null);
    }

    /**
     * Initializes the process-global hardware execution registry with a custom backend plugin path.
     *
     * @param customPluginPath explicit directory to search for dynamic ggml plugin libraries, or null to use compiled backends
     * @return true if backend registers successfully, false otherwise
     */
    public static synchronized boolean init(String customPluginPath) {
        if (initialized) {
            return true;
        }
        byte[] pathBytes = (customPluginPath != null && !customPluginPath.isEmpty())
            ? ArgusJniValidation.toUtf8NoNull(customPluginPath, "customPluginPath")
            : null;
        initialized = ArgusJni.nativeBackendInit(pathBytes);
        return initialized;
    }

    /**
     * Queries whether the process-global native compute backend registry is currently initialized.
     */
    public static boolean isInitialized() {
        return ArgusJni.nativeBackendIsInitialized();
    }

    /**
     * Queries the compile-time feature bitmask representing backends and acceleration built into the native binary.
     */
    public static long getBuildFeatures() {
        return ArgusJni.nativeBuildFeatures();
    }

    /**
     * Checks if a specific feature flag is present in the native library build.
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
        ArgusJni.setLogCallback(null);
        if (!ArgusJni.isInLogCallback()) {
            long deadline = System.currentTimeMillis() + 1000;
            while (System.currentTimeMillis() < deadline) {
                if (ArgusJni.nativeLogInFlightCount() == 0) {
                    break;
                }
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
        ArgusJni.nativeBackendFree();
        initialized = false;
    }

    /**
     * Sets the global minimum native log severity threshold.
     */
    public static void setLogLevel(ArgusLogLevel level) {
        if (level == null) {
            throw new IllegalArgumentException("Log level cannot be null");
        }
        ArgusJni.nativeSetLogLevel(level.getValue());
    }

    /**
     * Queries the active global minimum native log severity threshold.
     */
    public static ArgusLogLevel getLogLevel() {
        return ArgusLogLevel.fromValue(ArgusJni.nativeGetLogLevel());
    }

    /**
     * Convenience method to silence or restore standard warning logging.
     */
    public static void setQuiet(boolean quiet) {
        setLogLevel(quiet ? ArgusLogLevel.NONE : ArgusLogLevel.WARN);
    }

    /**
     * Registers a custom log handler callback for native diagnostic telemetry.
     */
    public static synchronized void setLogCallback(ArgusLogCallback callback) {
        ArgusJni.setLogCallback(callback);
    }

    /**
     * @return count of available hardware backends.
     */
    public static int getCount() {
        return ArgusJni.nativeBackendGetCount();
    }

    /**
     * Query driver registry name at index.
     */
    public static String getName(int index) {
        String name = ArgusJni.decodeUtf8(ArgusJni.nativeBackendGetName(index));
        return (name != null && !name.isEmpty()) ? name : "UNKNOWN";
    }

    /**
     * Utility listing all available backends on the machine.
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
        return ArgusJni.VERSION;
    }

    public static void main(String[] args) {
        String expectedTarget = args.length > 0 ? args[0] : null;
        System.out.println("[ArgusBackend-JNI] Version: " + getVersion());
        System.out.println("[ArgusBackend-JNI] Extracted/Loaded dir: " + ArgusJni.EXTRACTED_DIR);
        long features = getBuildFeatures();
        System.out.println("[ArgusBackend-JNI] Build features bitmask: 0x" + Long.toHexString(features));
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
        return switch (target.toLowerCase(Locale.ROOT)) {
            case "cpu" -> FEATURE_CPU;
            case "cuda" -> FEATURE_CPU | FEATURE_CUDA;
            case "rocm", "hip" -> FEATURE_CPU | FEATURE_ROCM;
            case "vulkan" -> FEATURE_CPU | FEATURE_VULKAN;
            case "metal" -> FEATURE_CPU | FEATURE_METAL;
            default -> throw new IllegalArgumentException("Unknown backend: " + target);
        };
    }

    /**
     * Build metadata only; does not initialize drivers or perform device execution.
     */
    public static Map<String, String> getBuildInfo() {
        String raw = ArgusJni.decodeUtf8(ArgusJni.nativeBuildInfo());
        if (raw == null || raw.isEmpty()) {
            throw new IllegalStateException("Cannot read native build metadata");
        }
        Map<String, String> info = new LinkedHashMap<>();
        for (String line : raw.split("\\n")) {
            if (line.isEmpty()) {
                continue;
            }
            int separator = line.indexOf('=');
            if (separator <= 0 || info.put(line.substring(0, separator), line.substring(separator + 1)) != null) {
                throw new IllegalStateException("Invalid build metadata");
            }
        }
        info.put("features", Long.toString(getBuildFeatures()));
        return Map.copyOf(info);
    }
}
