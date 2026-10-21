package cc.projectargus.libargus.platform.cpu;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class WindowsCpuNativeProvider extends AbstractNativeLibraryProvider {
    public WindowsCpuNativeProvider() {
        super("windows", "amd64", "cpu", 10, "argus.dll");
    }
}
