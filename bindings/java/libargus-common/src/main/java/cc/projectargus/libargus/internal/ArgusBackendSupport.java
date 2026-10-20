package cc.projectargus.libargus.internal;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Shared compile-time feature flag constants, expected target masks, and build-info metadata parsing
 * shared between {@code :libargus-core} and {@code :libargus-jni}.
 */
public final class ArgusBackendSupport {

    public static final long FEATURE_CPU_ACCEL = 1L << 0;
    public static final long FEATURE_CPU       = FEATURE_CPU_ACCEL;
    public static final long FEATURE_CUDA      = 1L << 1;
    public static final long FEATURE_ROCM      = 1L << 2;
    public static final long FEATURE_VULKAN    = 1L << 3;
    public static final long FEATURE_METAL     = 1L << 4;
    public static final long FEATURE_SYCL      = 1L << 5;

    private ArgusBackendSupport() {}

    /**
     * Resolves the expected compile-time feature bitmask for a target backend name.
     */
    public static long expectedFeatureMask(String target) {
        if (target == null) {
            throw new IllegalArgumentException("Backend target cannot be null");
        }
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
     * Parses newline-delimited {@code key=value} metadata returned by {@code argus_build_info_copy}.
     */
    public static Map<String, String> parseBuildInfo(String raw, long buildFeatures) {
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
        info.put("features", Long.toString(buildFeatures));
        return info;
    }

    /**
     * Executes the CLI diagnostic feature probe used by {@code ArgusBackend.main(String[])}.
     */
    public static void runCliProbe(String label, String version, String extractedDir, long features, String[] args) {
        String expectedTarget = (args != null && args.length > 0) ? args[0] : null;
        System.out.println("[" + label + "] Version: " + version);
        System.out.println("[" + label + "] Extracted/Loaded dir: " + extractedDir);
        System.out.println("[" + label + "] Build features bitmask: 0x" + Long.toHexString(features));
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
}
