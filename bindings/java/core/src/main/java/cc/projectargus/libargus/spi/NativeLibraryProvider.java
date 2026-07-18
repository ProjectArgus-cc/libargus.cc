package cc.projectargus.libargus.spi;

import java.io.InputStream;

/**
 * Service Provider Interface (SPI) for registering and loading native libargus binaries.
 */
public interface NativeLibraryProvider {
    /**
     * @return the operating system supported by this provider (e.g. "linux", "windows", "macos")
     */
    String getOs();

    /**
     * @return the CPU architecture supported by this provider (e.g. "amd64", "aarch64", "x86_64")
     */
    String getArch();

    /**
     * @return the GPU/compute acceleration backend type (e.g. "cpu", "cuda", "rocm", "sycl", "vulkan", "metal")
     */
    String getBackend();

    /**
     * @return priority of this provider (higher priorities are loaded first, e.g. CUDA=100, ROCm=90, CPU=10)
     */
    int getPriority();

    /**
     * Opens an input stream to read the native library binary resource.
     * 
     * @return resource input stream
     */
    InputStream getLibraryStream();

    /**
     * @return the platform-specific library name (e.g. "libargus.so", "argus.dll")
     */
    String getLibraryName();
}
