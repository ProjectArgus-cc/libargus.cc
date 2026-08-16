package cc.projectargus.libargus.platform.cpu;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class WindowsCpuNativeProvider implements NativeLibraryProvider {
    @Override
    public String getOs() {
        return "windows";
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
        return getClass().getResourceAsStream("/natives/windows-amd64/cpu/argus.dll");
    }

    @Override
    public String getLibraryName() {
        return "argus.dll";
    }
}
