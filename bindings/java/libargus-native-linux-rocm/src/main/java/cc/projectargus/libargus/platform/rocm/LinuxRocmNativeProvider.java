package cc.projectargus.libargus.platform.rocm;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class LinuxRocmNativeProvider extends AbstractNativeLibraryProvider {
    public LinuxRocmNativeProvider() {
        super("linux", "amd64", "rocm", 90, "libargus.so");
    }
}
