package cc.projectargus.libargus.platform.vulkan;

import cc.projectargus.libargus.spi.NativeLibraryProvider;
import java.io.InputStream;

public final class WindowsVulkanNativeProvider implements NativeLibraryProvider {
    @Override
    public String getOs() {
        return "windows";
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
        return getClass().getResourceAsStream("/natives/windows-amd64/vulkan/argus.dll");
    }

    @Override
    public String getLibraryName() {
        return "argus.dll";
    }
}
