package cc.projectargus.libargus.internal;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Abstract base class for all native unmanaged resources in libargus.
 * Enforces zero-allocation native handle leasing via {@link ReentrantReadWriteLock}
 * and guarantees thread-safe, idempotent lifecycle state transitions.
 */
public abstract class ArgusNativeResource implements AutoCloseable {
    protected final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    protected volatile MemorySegment handle;
    protected volatile boolean closed = false;

    protected ArgusNativeResource(MemorySegment handle) {
        this.handle = Objects.requireNonNull(handle, "handle cannot be null");
    }

    /**
     * Acquires a shared read lease on this native resource.
     * Prevents concurrent closing while native operations execute off-heap.
     * Must be paired with {@link #releaseReadLease()} in a finally block.
     *
     * @return the valid native handle segment
     * @throws IllegalStateException if the resource is closed or the handle is NULL
     */
    public MemorySegment acquireReadLease() {
        lifecycleLock.readLock().lock();
        if (closed || handle == null) {
            lifecycleLock.readLock().unlock();
            throw new IllegalStateException(resourceName() + " is already closed");
        }
        return handle;
    }

    /**
     * Releases a previously acquired shared read lease.
     */
    public void releaseReadLease() {
        lifecycleLock.readLock().unlock();
    }

    /**
     * Returns true if this resource has been closed.
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Accessor for the underlying native memory address.
     *
     * @return the memory segment handle
     * @throws IllegalStateException if this resource is closed
     */
    public MemorySegment getHandle() {
        if (closed || handle == null) {
            throw new IllegalStateException(resourceName() + " is already closed");
        }
        return handle;
    }

    /**
     * Returns the human-readable class/resource name for diagnostic error reporting.
     */
    protected abstract String resourceName();

    /**
     * Executes native deallocation logic for this resource under write lock exclusion.
     *
     * @param oldHandle the valid native pointer that was held prior to closure
     */
    protected abstract void releaseNative(MemorySegment oldHandle);

    @Override
    public void close() {
        if (closed) {
            return;
        }
        lifecycleLock.writeLock().lock();
        MemorySegment oldHandle;
        try {
            if (closed) {
                return;
            }
            closed = true;
            oldHandle = handle;
            handle = MemorySegment.NULL;
            if (oldHandle != null && !oldHandle.equals(MemorySegment.NULL)) {
                releaseNative(oldHandle);
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
