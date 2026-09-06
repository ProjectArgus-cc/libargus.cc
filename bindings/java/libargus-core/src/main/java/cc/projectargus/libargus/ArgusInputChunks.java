package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Holds tokenized prompt chunk lists in unmanaged native memory.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusInputChunks implements AutoCloseable {
    private MemorySegment chunksPtr;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private ArgusInputChunks(MemorySegment chunksPtr) {
        this.chunksPtr = Objects.requireNonNull(chunksPtr);
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

    public MemorySegment getHandle() {
        checkNotClosed();
        return chunksPtr;
    }

    public boolean isClosed() {
        return closed.get();
    }

    private void checkNotClosed() {
        if (closed.get() || chunksPtr == null || chunksPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("ArgusInputChunks container has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (chunksPtr != null && !chunksPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_input_chunks_free.invokeExact(chunksPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native input chunks resources", t);
            } finally {
                chunksPtr = MemorySegment.NULL;
            }
        }
    }
}
