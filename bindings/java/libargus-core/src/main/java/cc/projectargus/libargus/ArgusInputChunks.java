package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import java.lang.foreign.MemorySegment;

/**
 * Holds tokenized prompt chunk lists in unmanaged native memory.
 * Extends {@link ArgusNativeResource} for safe native leasing and idempotent lifecycle deallocation.
 */
public final class ArgusInputChunks extends ArgusNativeResource {

    private ArgusInputChunks(MemorySegment chunksPtr) {
        super(chunksPtr);
    }

    @Override
    protected String resourceName() {
        return "ArgusInputChunks";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_input_chunks_free.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        }
    }

    /**
     * Allocates an empty input chunks container list.
     */
    public static ArgusInputChunks init() {
        try {
            MemorySegment chunksPtr = (MemorySegment) ArgusBindings.argus_input_chunks_init.invokeExact();
            if (chunksPtr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_input_chunks_init");
            }
            return new ArgusInputChunks(chunksPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to allocate unmanaged input chunks container", t);
        }
    }
}
