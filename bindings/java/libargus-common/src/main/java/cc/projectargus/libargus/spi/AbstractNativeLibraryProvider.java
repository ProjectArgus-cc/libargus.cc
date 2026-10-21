package cc.projectargus.libargus.spi;

import java.io.InputStream;
import java.util.Objects;

/**
 * Base implementation of {@link NativeLibraryProvider} encapsulating platform metadata
 * and classpath resource resolution for packaged native runtime classifiers.
 */
public abstract class AbstractNativeLibraryProvider implements NativeLibraryProvider {
    private final String os;
    private final String arch;
    private final String backend;
    private final int priority;
    private final String libraryName;
    private final String resourcePath;

    protected AbstractNativeLibraryProvider(String os, String arch, String backend, int priority, String libraryName) {
        this.os = Objects.requireNonNull(os, "os");
        this.arch = Objects.requireNonNull(arch, "arch");
        this.backend = Objects.requireNonNull(backend, "backend");
        this.priority = priority;
        this.libraryName = Objects.requireNonNull(libraryName, "libraryName");
        this.resourcePath = "/natives/" + os + "-" + arch + "/" + backend + "/" + libraryName;
    }

    @Override
    public final String getOs() {
        return os;
    }

    @Override
    public final String getArch() {
        return arch;
    }

    @Override
    public final String getBackend() {
        return backend;
    }

    @Override
    public final int getPriority() {
        return priority;
    }

    @Override
    public final String getLibraryName() {
        return libraryName;
    }

    @Override
    public InputStream getLibraryStream() {
        return getClass().getResourceAsStream(resourcePath);
    }
}
