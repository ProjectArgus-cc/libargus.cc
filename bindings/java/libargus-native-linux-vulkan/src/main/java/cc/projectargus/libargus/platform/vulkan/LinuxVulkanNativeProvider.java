package cc.projectargus.libargus.platform.vulkan;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class LinuxVulkanNativeProvider extends AbstractNativeLibraryProvider {
    public LinuxVulkanNativeProvider() {
        super("linux", "amd64", "vulkan", 80, "libargus.so");
    }
}
