package cc.projectargus.libargus.platform.rocm;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

/**
 * Native Library Provider for ROCm/HIP execution.
 */
public final class RocmNativeProvider implements NativeLibraryProvider {
    @Override
    public String getOs() {
        return System.getProperty("os.name").toLowerCase();
    }

    @Override
    public String getArch() {
        return System.getProperty("os.arch").toLowerCase();
    }

    @Override
    public String getBackend() {
        return "rocm";
    }

    @Override
    public int getPriority() {
        return 90; // ROCm is high priority, just under CUDA
    }

    @Override
    public InputStream getLibraryStream() {
        String osName = getOs();
        String osArch = getArch();
        String osDir;
        if (osName.contains("linux")) {
            osDir = "linux-" + osArch;
        } else if (osName.contains("windows")) {
            osDir = "windows-" + osArch;
        } else if (osName.contains("mac")) {
            osDir = "macos-" + osArch;
        } else {
            osDir = "unknown";
        }
        String libName = getLibraryName();
        return getClass().getResourceAsStream("/natives/" + osDir + "/rocm/" + libName);
    }

    @Override
    public String getLibraryName() {
        String osName = getOs();
        if (osName.contains("windows")) {
            return "argus.dll";
        } else if (osName.contains("mac")) {
            return "libargus.dylib";
        } else {
            return "libargus.so";
        }
    }
}
