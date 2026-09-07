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
     * Scoped RAII token guaranteeing that the underlying native resource
     * remains leased and valid until {@link #close()} is called.
     */
    public final class Lease implements AutoCloseable {
        private final MemorySegment leasedHandle;
        private boolean released = false;

        private Lease(MemorySegment leasedHandle) {
            this.leasedHandle = leasedHandle;
        }

        /**
         * Access the valid, leased native handle.
         *
         * @return valid native MemorySegment
         * @throws IllegalStateException if this lease has already been closed
         */
        public MemorySegment handle() {
            if (released) {
                throw new IllegalStateException("Lease for " + resourceName() + " has already been closed");
            }
            return leasedHandle;
        }

        @Override
        public void close() {
            if (!released) {
                released = true;
                releaseReadLease();
            }
        }
    }

    /**
     * Acquires a scoped read lease on this native resource.
     * Prevents concurrent {@link #close()} until the returned {@link Lease} is closed.
     *
     * @return an active AutoCloseable {@link Lease}
     * @throws IllegalStateException if the resource is closed
     */
    public Lease lease() {
        return new Lease(acquireReadLease());
    }

    /**
     * Functional callback action executing over a leased native handle.
     */
    @FunctionalInterface
    public interface ResourceAction<R, E extends Throwable> {
        R execute(MemorySegment handle) throws E;
    }

    /**
     * Executes the given action under a shared read lease, guaranteeing balanced acquisition and release.
     *
     * @param action callback to invoke with leased handle
     * @param <R> return type
     * @param <E> exception type
     * @return result of the action
     * @throws E if action throws
     * @throws IllegalStateException if resource is closed
     */
    public <R, E extends Throwable> R withHandle(ResourceAction<R, E> action) throws E {
        MemorySegment h = acquireReadLease();
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
     *
     * @return the valid native handle segment
     * @throws IllegalStateException if the resource is closed or the handle is NULL
     */
    protected MemorySegment acquireReadLease() {
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
     * Accessor for the unmanaged native memory address without acquiring a lease.
     * <p>
     * <b>WARNING:</b> This returns an unleased borrowed pointer that is NOT protected
     * against concurrent closure or use-after-free. If another thread calls {@link #close()},
     * this memory address may be deallocated immediately. Use {@link #lease()} or standard
     * high-level methods instead.
     *
     * @return the raw native memory segment handle
     * @throws IllegalStateException if this resource is closed
     */
    public MemorySegment unsafeBorrowedHandle() {
        if (closed || handle == null) {
            throw new IllegalStateException(resourceName() + " is already closed");
        }
        return handle;
    }

    /**
     * Deprecated accessor for unmanaged native memory address.
     *
     * @deprecated Use {@link #lease()} or {@link #unsafeBorrowedHandle()} instead.
     * @return the memory segment handle
     * @throws IllegalStateException if this resource is closed
     */
    @Deprecated
    public MemorySegment getHandle() {
        return unsafeBorrowedHandle();
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
