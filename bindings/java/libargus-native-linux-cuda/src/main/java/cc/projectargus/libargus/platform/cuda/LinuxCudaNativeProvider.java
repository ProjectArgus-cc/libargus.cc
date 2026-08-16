package cc.projectargus.libargus.platform.cuda;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class LinuxCudaNativeProvider implements NativeLibraryProvider {
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
        return "cuda";
    }

    @Override
    public int getPriority() {
        return 100;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream("/natives/linux-amd64/cuda/libargus.so");
    }

    @Override
    public String getLibraryName() {
        return "libargus.so";
    }
}
