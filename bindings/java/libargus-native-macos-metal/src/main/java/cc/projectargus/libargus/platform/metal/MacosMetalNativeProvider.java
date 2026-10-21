package cc.projectargus.libargus.platform.metal;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class MacosMetalNativeProvider extends AbstractNativeLibraryProvider {
    public MacosMetalNativeProvider() {
        super("macos", "aarch64", "metal", 100, "libargus.dylib");
    }
}
