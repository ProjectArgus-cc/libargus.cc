package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wraps loaded GGUF model weights in unmanaged memory.
 * Implements AutoCloseable to ensure resources are cleaned up safely.
 */
public final class ArgusModel implements AutoCloseable {
    private MemorySegment modelPtr;
    private final java.util.concurrent.atomic.AtomicInteger refCount = new java.util.concurrent.atomic.AtomicInteger(1);

    ArgusModel(MemorySegment modelPtr) {
        this.modelPtr = Objects.requireNonNull(modelPtr);
    }

    void acquire() {
        int current;
        do {
            current = refCount.get();
            if (current <= 0) {
                throw new IllegalStateException("Cannot acquire reference; ArgusModel has already been closed");
            }
        } while (!refCount.compareAndSet(current, current + 1));
    }

    void release() {
        if (refCount.decrementAndGet() == 0) {
            freeNativeModel();
        }
    }

    int getRefCount() {
        return refCount.get();
    }

    void clearHandleForTesting() {
        this.modelPtr = MemorySegment.NULL;
    }

    private synchronized void freeNativeModel() {
        if (modelPtr != null && !modelPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_model_free.invokeExact(modelPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to free native model resources", t);
            } finally {
                modelPtr = MemorySegment.NULL;
            }
        }
    }

    /**
     * Loads GGUF weights off-heap using the specified configuration parameters.
     *
     * @param arena     the allocation scope for the temporary parameter structures
     * @param modelPath path to the local GGUF file
     * @param gpuLayers number of layers to offload to VRAM (RTX 4090 etc.)
     * @param useMlock  force locking pages to RAM to prevent swapping
     * @return a loaded ArgusModel wrapper
     */
    public static ArgusModel load(Arena arena, Path modelPath, int gpuLayers, boolean useMlock) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(modelPath);

        // Allocate local path string off-heap
        MemorySegment pathSegment = arena.allocateFrom(modelPath.toAbsolutePath().toString());

        // Allocate the argus_model_params struct off-heap
        MemorySegment paramsSegment = arena.allocate(ArgusLayouts.MODEL_PARAMS);

        // Populate struct fields using offset layout mappings
        paramsSegment.set(ValueLayout.ADDRESS, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("model_path")), 
            pathSegment
        );
        paramsSegment.set(ValueLayout.JAVA_INT, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("gpu_layers")), 
            gpuLayers
        );
        paramsSegment.set(ValueLayout.JAVA_BOOLEAN, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("use_mlock")), 
            useMlock
        );

        try {
            MemorySegment modelPtr = (MemorySegment) ArgusBindings.argus_model_load.invokeExact(paramsSegment);
            if (modelPtr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_model_load returned NULL for: " + modelPath);
            }
            return new ArgusModel(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute native load for " + modelPath, t);
        }
    }

    /**
     * Returns the raw memory address representing the unmanaged model structure.
     */
    public MemorySegment getHandle() {
        return modelPtr;
    }

    @Override
    public void close() {
        release();
    }
}
