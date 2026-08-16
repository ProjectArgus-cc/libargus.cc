package cc.projectargus.libargus.platform.rocm;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class LinuxRocmNativeProvider implements NativeLibraryProvider {
    @Override
    public String getOs() {
        return "linux";
    }

    @Override
    public String getArch() {
        return "amd64";
    }

    @Override
    public String getBackend() {
        return "rocm";
    }

    @Override
    public int getPriority() {
        return 90;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream("/natives/linux-amd64/rocm/libargus.so");
    }

    @Override
    public String getLibraryName() {
        return "libargus.so";
    }
}
