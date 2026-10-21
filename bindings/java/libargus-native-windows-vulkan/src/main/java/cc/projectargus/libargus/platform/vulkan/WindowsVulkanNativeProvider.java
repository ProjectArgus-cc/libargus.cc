package cc.projectargus.libargus.platform.vulkan;

import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;

public final class WindowsVulkanNativeProvider extends AbstractNativeLibraryProvider {
    public WindowsVulkanNativeProvider() {
        super("windows", "amd64", "vulkan", 80, "argus.dll");
    }
}
