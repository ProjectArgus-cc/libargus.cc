package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Handles the process-global compute backends (CUDA, CPU, Metal) registry and telemetry.
 */
public final class ArgusBackend {
    private static boolean initialized = false;

    private ArgusBackend() {}

    public static synchronized boolean init() {
        return init(ArgusBindings.EXTRACTED_DIR);
    }

    /**
     * Initializes the process-global hardware execution registry with a custom backend plugin path.
     * Must be called once before loading models or executing transcription/speech.
     *
     * @param customPluginPath directory path to search for dynamic ggml plugin libraries (like CUDA)
     * @return true if backend registers successfully, false otherwise.
     */
    public static synchronized boolean init(String customPluginPath) {
        if (initialized) {
            return true;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment pathSeg = (customPluginPath != null && !customPluginPath.isEmpty())
                ? localArena.allocateFrom(customPluginPath)
                : MemorySegment.NULL;
            initialized = (boolean) ArgusBindings.argus_backend_init.invokeExact(pathSeg);
            return initialized;
        } catch (Throwable t) {
            throw new RuntimeException("Fatal error running argus_backend_init", t);
        }
    }

    /**
     * Deallocates global backend context registration and frees active compute pools.
     */
    public static synchronized void free() {
        if (!initialized) {
            return;
        }
        try {
            ArgusBindings.argus_backend_free.invokeExact();
            initialized = false;
        } catch (Throwable t) {
            throw new RuntimeException("Fatal error running argus_backend_free", t);
        }
    }

    /**
     * @return count of available hardware backends.
     */
    public static int getCount() {
        try {
            return (int) ArgusBindings.argus_backend_get_count.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("Fatal error running argus_backend_get_count", t);
        }
    }

    /**
     * Query driver registry name at index.
     * @param index backend index.
     * @return dynamic backend name (e.g. "CUDA0", "CPU").
     */
    public static String getName(int index) {
        try {
            MemorySegment namePtr = (MemorySegment) ArgusBindings.argus_backend_get_name.invokeExact(index);
            if (namePtr.equals(MemorySegment.NULL)) {
                return "UNKNOWN";
            }
            return namePtr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Fatal error running argus_backend_get_name", t);
        }
    }

    /**
     * Utility listing all available backends on the machine.
     * @return list of backend driver names.
     */
    public static List<String> getAvailableBackends() {
        int count = getCount();
        if (count <= 0) {
            return Collections.emptyList();
        }
        List<String> list = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            list.add(getName(i));
        }
        return Collections.unmodifiableList(list);
    }
}
