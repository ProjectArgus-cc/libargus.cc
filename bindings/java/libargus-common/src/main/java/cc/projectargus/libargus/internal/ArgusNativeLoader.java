package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Unified native library discovery, SPI extraction, and loading engine shared across
 * Project Panama FFM (:libargus-core) and JNI (:libargus-jni) bindings.
 *
 * Resolution hierarchy:
 * 1. Explicit override property: -Dcc.projectargus.libargus.path=/abs/path/to/libargus.so
 * 2. ServiceLoader<NativeLibraryProvider> SPI auto-extraction from :libargus-native-* JARs
 * 3. Standard OS/JVM library path: System.loadLibrary("argus") (supporting Android APK jniLibs)
 * 4. Bounded local workspace build/ directory search (depth <= 3)
 */
public final class ArgusNativeLoader {
    public static final String VERSION;
    public static final String EXTRACTED_DIR;
    private static final String LOADED_LIBRARY_PATH;
    private static final String LOADED_PROVIDER;

    static {
        String resolvedVersion = "0.0.0";
        try (InputStream is = ArgusNativeLoader.class.getResourceAsStream("/version.txt")) {
            if (is != null) {
                resolvedVersion = new String(is.readAllBytes(), StandardCharsets.UTF_8).trim();
            }
        } catch (IOException ignored) {
        }
        VERSION = resolvedVersion;

        String resolvedExtractedDir = null;
        String resolvedLibPath = null;
        String resolvedProvider = null;

        String customPath = System.getProperty("cc.projectargus.libargus.path");
        boolean verifySpi = Boolean.getBoolean("cc.projectargus.libargus.verifySpi");
        if (verifySpi && customPath != null) {
            throw new IllegalStateException("Verifier prohibits a native path override");
        }

        if (customPath != null && !customPath.trim().isEmpty()) {
            Path path = Paths.get(customPath.trim()).toAbsolutePath();
            if (Files.isDirectory(path)) {
                path = path.resolve(System.mapLibraryName("argus"));
            }
            System.load(path.toString());
            resolvedLibPath = path.toString();
            resolvedExtractedDir = path.getParent() != null ? path.getParent().toString() : null;
        } else {
            SpiLoadResult spiResult = tryLoadFromSPI(resolvedVersion);
            if (spiResult != null) {
                resolvedExtractedDir = spiResult.extractedDir;
                resolvedLibPath = spiResult.libraryPath;
                resolvedProvider = spiResult.providerName;
            } else {
                if (verifySpi) {
                    throw new UnsatisfiedLinkError("Verifier requires exactly one working SPI provider");
                }
                boolean loaded = false;
                try {
                    System.loadLibrary("argus");
                    loaded = true;
                } catch (UnsatisfiedLinkError ignored) {
                    // Fall back to bounded local workspace build directory search
                }

                if (!loaded) {
                    String userDir = System.getProperty("user.dir");
                    if (userDir != null) {
                        String libName = System.mapLibraryName("argus");
                        File currentDir = new File(userDir).getAbsoluteFile();
                        int maxDepth = 3;
                        while (currentDir != null && maxDepth-- >= 0) {
                            File buildDir = new File(currentDir, "build");
                            File localLibInLib = new File(new File(buildDir, "lib"), libName);
                            File localLibInBin = new File(new File(buildDir, "bin"), libName);
                            File localLibInLibRel = new File(new File(new File(buildDir, "lib"), "Release"), libName);
                            File localLibInBinRel = new File(new File(new File(buildDir, "bin"), "Release"), libName);
                            File localLib = new File(buildDir, libName);

                            File targetLib = null;
                            if (localLibInLib.exists()) {
                                targetLib = localLibInLib;
                            } else if (localLibInBin.exists()) {
                                targetLib = localLibInBin;
                            } else if (localLibInLibRel.exists()) {
                                targetLib = localLibInLibRel;
                            } else if (localLibInBinRel.exists()) {
                                targetLib = localLibInBinRel;
                            } else if (localLib.exists()) {
                                targetLib = localLib;
                            }

                            if (targetLib != null) {
                                System.load(targetLib.getAbsolutePath());
                                resolvedLibPath = targetLib.getAbsolutePath();
                                resolvedExtractedDir = targetLib.getParent();
                                loaded = true;
                                break;
                            }
                            currentDir = currentDir.getParentFile();
                        }
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
        LOADED_LIBRARY_PATH = resolvedLibPath;
        LOADED_PROVIDER = resolvedProvider;
    }

    private ArgusNativeLoader() {}

    public static void ensureLoaded() {
        // Triggers static initializer if not already executed
    }

    public static String loadedLibraryPath() {
        return LOADED_LIBRARY_PATH;
    }

    public static String loadedProvider() {
        return LOADED_PROVIDER;
    }

    private static final class SpiLoadResult {
        final String extractedDir;
        final String libraryPath;
        final String providerName;

        SpiLoadResult(String extractedDir, String libraryPath, String providerName) {
            this.extractedDir = extractedDir;
            this.libraryPath = libraryPath;
            this.providerName = providerName;
        }
    }

    private static SpiLoadResult tryLoadFromSPI(String version) {
        String osName = System.getProperty("os.name", "").toLowerCase();
        String osArch = System.getProperty("os.arch", "").toLowerCase();

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
            boolean archMatch =
                (normalizedArch.equals("amd64") && (pArch.equals("amd64") || pArch.equals("x86_64") || pArch.equals("x64"))) ||
                (normalizedArch.equals("aarch64") && (pArch.equals("aarch64") || pArch.equals("arm64"))) ||
                pArch.equals(normalizedArch);

            if (osMatch && archMatch) {
                providers.add(provider);
            }
        }

        if (Boolean.getBoolean("cc.projectargus.libargus.verifySpi") && providers.size() != 1) {
            throw new IllegalStateException("Verifier requires exactly one matching provider");
        }
        if (providers.isEmpty()) {
            return null;
        }

        providers.sort(Comparator.comparingInt(NativeLibraryProvider::getPriority).reversed());

        String customDir = System.getProperty("cc.projectargus.libargus.nativeDir");
        if (customDir == null || customDir.trim().isEmpty()) {
            customDir = System.getenv("LIBARGUS_NATIVE_DIR");
        }
        File destDir;
        if (customDir != null && !customDir.trim().isEmpty()) {
            destDir = new File(customDir.trim());
        } else {
            String rawUser = System.getProperty("user.name", "default");
            String safeUser = rawUser.replaceAll("[^a-zA-Z0-9._-]", "_");
            destDir = new File(System.getProperty("java.io.tmpdir"), "argus_native_cache_" + safeUser);
        }

        try {
            if (!destDir.exists()) {
                if (FileSystems.getDefault().supportedFileAttributeViews().contains("posix")) {
                    Files.createDirectories(
                        destDir.toPath(),
                        PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------"))
                    );
                } else if (!destDir.mkdirs() && !destDir.exists()) {
                    throw new RuntimeException("Failed to create native extraction directory: " + destDir);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize secure native extraction directory: " + destDir, e);
        }

        for (NativeLibraryProvider provider : providers) {
            String libVersionName = System.mapLibraryName("argus-" + provider.getBackend() + "-" + version);
            File targetFile = new File(destDir, libVersionName);

            try {
                if (targetFile.exists() && targetFile.length() > 0) {
                    System.load(targetFile.getAbsolutePath());
                    return new SpiLoadResult(
                        destDir.getAbsolutePath(),
                        targetFile.getAbsolutePath(),
                        provider.getClass().getName()
                    );
                }

                File tempFile = File.createTempFile("libargus_extract_", ".tmp", destDir);
                try {
                    try (InputStream is = provider.getLibraryStream()) {
                        if (is == null) {
                            continue;
                        }
                        Files.copy(is, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    }
                    Files.move(
                        tempFile.toPath(),
                        targetFile.toPath(),
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING
                    );
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
                return new SpiLoadResult(
                    destDir.getAbsolutePath(),
                    targetFile.getAbsolutePath(),
                    provider.getClass().getName()
                );
            } catch (UnsatisfiedLinkError | Exception e) {
                System.err.println("Warning: Failed to load native library from provider [" +
                    provider.getClass().getName() + " | Backend: " + provider.getBackend() + "]: " + e.getMessage());
            }
        }

        return null;
    }
}
