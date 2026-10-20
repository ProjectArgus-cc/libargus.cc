package cc.projectargus.libargus.internal;

import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Generic abstract base class for all native unmanaged resources in libargus.
 * Enforces handle leasing via {@link ReentrantReadWriteLock}, thread-affinity lease release,
 * self-upgrade deadlock prevention, and idempotent write-locked native deallocation.
 *
 * @param <H> Native handle representation (e.g. MemorySegment in FFM or Long in JNI)
 */
public abstract class AbstractNativeResource<H> implements AutoCloseable {
    protected final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    protected volatile H handle;
    protected volatile boolean closed = false;

    protected AbstractNativeResource(H handle) {
        this.handle = Objects.requireNonNull(handle, "handle cannot be null");
    }

    /**
     * Returns the sentinel value representing a null/cleared native handle.
     */
    protected abstract H nullHandle();

    /**
     * Checks whether the given handle is null or equivalent to the cleared sentinel.
     */
    protected abstract boolean isNullHandle(H handle);

    /**
     * Returns the human-readable class/resource name for diagnostic error reporting.
     */
    protected abstract String resourceName();

    /**
     * Executes native deallocation logic for this resource under write lock exclusion.
     *
     * @param oldHandle the valid native handle that was held prior to closure
     */
    protected abstract void releaseNative(H oldHandle);

    /**
     * Functional callback action executing over a leased native handle.
     */
    @FunctionalInterface
    public interface HandleAction<H, R, E extends Throwable> {
        R execute(H handle) throws E;
    }

    /**
     * Executes the given action under a shared read lease, guaranteeing balanced acquisition and release.
     */
    protected <R, E extends Throwable> R withHandle(HandleAction<H, R, E> action) throws E {
        H h = acquireReadLease();
        try {
            return action.execute(h);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Acquires a shared read lease on this native resource.
     * Prevents concurrent closing while native operations execute off-heap.
     * Must be paired with {@link #releaseReadLease()} in a finally block.
     */
    protected H acquireReadLease() {
        lifecycleLock.readLock().lock();
        H current = handle;
        if (closed || current == null) {
            lifecycleLock.readLock().unlock();
            throw new IllegalStateException(resourceName() + " is already closed");
        }
        return current;
    }

    /**
     * Releases a previously acquired shared read lease.
     */
    protected void releaseReadLease() {
        lifecycleLock.readLock().unlock();
    }

    /**
     * Returns true if this resource has been closed.
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Accessor for the unmanaged native handle without acquiring a lease.
     */
    public H unsafeBorrowedHandle() {
        H current = handle;
        if (closed || current == null) {
            throw new IllegalStateException(resourceName() + " is already closed");
        }
        return current;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (lifecycleLock.getReadHoldCount() > 0 && !lifecycleLock.isWriteLockedByCurrentThread()) {
            throw new IllegalStateException("Cannot close " + resourceName() +
                " while the current thread holds an active lease (self-upgrade deadlock prevention)");
        }
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            H oldHandle = handle;
            handle = nullHandle();
            if (!isNullHandle(oldHandle)) {
                releaseNative(oldHandle);
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
