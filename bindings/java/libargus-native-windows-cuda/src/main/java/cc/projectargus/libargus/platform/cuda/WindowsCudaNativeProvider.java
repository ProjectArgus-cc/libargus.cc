package cc.projectargus.libargus.platform.cuda;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class WindowsCudaNativeProvider extends AbstractNativeLibraryProvider {
    public WindowsCudaNativeProvider() {
        super("windows", "amd64", "cuda", 100, "argus.dll");
    }
}
