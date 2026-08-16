package cc.projectargus.libargus.platform.vulkan;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class LinuxVulkanNativeProvider implements NativeLibraryProvider {
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
        return "vulkan";
    }

    @Override
    public int getPriority() {
        return 80;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream("/natives/linux-amd64/vulkan/libargus.so");
    }

    @Override
    public String getLibraryName() {
        return "libargus.so";
    }
}
