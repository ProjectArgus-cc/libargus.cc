package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import java.lang.foreign.MemorySegment;

/**
 * A thread-safe native atomic cancellation object (argus_abort_flag_t)
 * to signal early termination to long-running native prefill and decode loops.
 * Extends {@link ArgusNativeResource} for safe native leasing and idempotent lifecycle deallocation.
 */
public final class ArgusAbortFlag extends ArgusNativeResource {

    /**
     * Initializes a new native atomic abort flag.
     */
    public ArgusAbortFlag() {
        super(createNativeHandle());
    }

    private static MemorySegment createNativeHandle() {
        try {
            MemorySegment h = (MemorySegment) ArgusBindings.argus_abort_flag_create.invokeExact();
            if (h == null || h.equals(MemorySegment.NULL)) {
                throw new ArgusNativeException(2, "Failed to allocate native argus_abort_flag_t");
            }
            return h;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Native abort flag instantiation failed", t);
        }
    }

    @Override
    protected String resourceName() {
        return "ArgusAbortFlag";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_abort_flag_release.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        }
    }

    /**
     * Signals cancellation to any active or subsequent native execution loops.
     */
    public void abort() {
        MemorySegment h = acquireReadLease();
        try {
            ArgusBindings.argus_abort_flag_request.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to signal native abort flag", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Resets the cancellation flag back to false.
     */
    public void reset() {
        MemorySegment h = acquireReadLease();
        try {
            ArgusBindings.argus_abort_flag_reset.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to reset native abort flag", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Checks if cancellation has been requested.
     */
    public boolean isAborted() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_abort_flag_is_requested.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query native abort flag status", t);
        } finally {
            releaseReadLease();
        }
    }
}
