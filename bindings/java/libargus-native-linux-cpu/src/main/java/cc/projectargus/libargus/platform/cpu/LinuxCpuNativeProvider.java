package cc.projectargus.libargus.platform.cpu;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class LinuxCpuNativeProvider extends AbstractNativeLibraryProvider {
    public LinuxCpuNativeProvider() {
        super("linux", "amd64", "cpu", 10, "libargus.so");
    }
}
