package cc.projectargus.libargus;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A safe, thread-safe cancellation flag encapsulating an unmanaged memory segment
 * to signal early termination to long-running native prefill loops.
 * Implements AutoCloseable to clean up its internal Arena.
 */
public final class ArgusAbortFlag implements AutoCloseable {
    private final Arena arena;
    private final MemorySegment handle;

    /**
     * Initializes a new ArgusAbortFlag with a default value of 0 (do not abort).
     */
    public ArgusAbortFlag() {
        this.arena = Arena.ofShared(); // Shared arena permits cross-thread access and modification
        this.handle = arena.allocate(ValueLayout.JAVA_INT);
        this.handle.set(ValueLayout.JAVA_INT, 0, 0);
    }

    /**
     * Writes 1 to the cancellation flag, signaling the native loop to abort execution.
     * This operation is thread-safe and can be called from any control thread.
     */
    public void abort() {
        handle.set(ValueLayout.JAVA_INT, 0, 1);
    }

    /**
     * Resets the cancellation flag back to 0.
     */
    public void reset() {
        handle.set(ValueLayout.JAVA_INT, 0, 0);
    }

    /**
     * Checks if the flag has been set to abort.
     *
     * @return true if the abort signal is active
     */
    public boolean isAborted() {
        return handle.get(ValueLayout.JAVA_INT, 0) != 0;
    }

    /**
     * Retrieves the underlying unmanaged memory segment handle.
     *
     * @return the unmanaged MemorySegment pointer
     */
    public MemorySegment getHandle() {
        return handle;
    }

    @Override
    public void close() {
        arena.close();
    }
}
