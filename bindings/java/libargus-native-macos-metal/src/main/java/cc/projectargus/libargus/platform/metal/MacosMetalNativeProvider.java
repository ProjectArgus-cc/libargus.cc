package cc.projectargus.libargus.platform.metal;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class MacosMetalNativeProvider implements NativeLibraryProvider {
    @Override
    public String getOs() {
        return "macos";
    }

    @Override
    public String getArch() {
        return "aarch64";
    }

    @Override
    public String getBackend() {
        return "metal";
    }

    @Override
    public int getPriority() {
        return 100;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream("/natives/macos-aarch64/metal/libargus.dylib");
    }

    @Override
    public String getLibraryName() {
        return "libargus.dylib";
    }
}
