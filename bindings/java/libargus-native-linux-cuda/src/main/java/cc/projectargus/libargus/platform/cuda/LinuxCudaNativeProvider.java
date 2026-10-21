package cc.projectargus.libargus.platform.cuda;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class LinuxCudaNativeProvider extends AbstractNativeLibraryProvider {
    public LinuxCudaNativeProvider() {
        super("linux", "amd64", "cuda", 100, "libargus.so");
    }
}
