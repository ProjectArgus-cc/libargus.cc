package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A thread-safe native reference-counted atomic cancellation object (argus_abort_flag_t)
 * to signal early termination to long-running native prefill and decode loops.
 * Implements AutoCloseable with idempotent release.
 */
public final class ArgusAbortFlag implements AutoCloseable {
    private final MemorySegment handle;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Initializes a new native reference-counted atomic abort flag.
     */
    public ArgusAbortFlag() {
        try {
            this.handle = (MemorySegment) ArgusBindings.argus_abort_flag_create.invokeExact();
            if (this.handle == null || this.handle.equals(MemorySegment.NULL)) {
                throw new ArgusNativeException(2, "Failed to allocate native argus_abort_flag_t");
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Native abort flag instantiation failed", t);
        }
    }

    /**
     * Signals cancellation to any active or subsequent native execution loops.
     */
    public void abort() {
        checkNotClosed();
        try {
            ArgusBindings.argus_abort_flag_request.invokeExact(handle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to signal native abort flag", t);
        }
    }

    /**
     * Resets the cancellation flag back to false.
     */
    public void reset() {
        checkNotClosed();
        try {
            ArgusBindings.argus_abort_flag_reset.invokeExact(handle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to reset native abort flag", t);
        }
    }

    /**
     * Checks if cancellation has been requested.
     */
    public boolean isAborted() {
        checkNotClosed();
        try {
            return (boolean) ArgusBindings.argus_abort_flag_is_requested.invokeExact(handle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to query native abort flag status", t);
        }
    }

    /**
     * Retrieves the underlying unmanaged memory segment handle.
     */
    public MemorySegment getHandle() {
        checkNotClosed();
        return handle;
    }

    public boolean isClosed() {
        return closed.get();
    }

    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("ArgusAbortFlag has already been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            try {
                ArgusBindings.argus_abort_flag_release.invokeExact(handle);
            } catch (Throwable t) {
                // Suppress destruction errors
            }
        }
    }
}
