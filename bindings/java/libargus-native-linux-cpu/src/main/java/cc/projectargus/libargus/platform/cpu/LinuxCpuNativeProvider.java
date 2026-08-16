package cc.projectargus.libargus.platform.cpu;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class LinuxCpuNativeProvider implements NativeLibraryProvider {
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
        return "cpu";
    }

    @Override
    public int getPriority() {
        return 10;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream("/natives/linux-amd64/cpu/libargus.so");
    }

    @Override
    public String getLibraryName() {
        return "libargus.so";
    }
}
