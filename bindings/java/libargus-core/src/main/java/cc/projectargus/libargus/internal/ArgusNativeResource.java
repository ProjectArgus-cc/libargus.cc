package cc.projectargus.libargus.internal;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Abstract base class for Project Panama FFM native resources in libargus.
 * Extends {@link AbstractNativeResource} with {@link MemorySegment} handles.
 */
public abstract class ArgusNativeResource extends AbstractNativeResource<MemorySegment> {

    protected ArgusNativeResource(MemorySegment handle) {
        super(Objects.requireNonNull(handle, "handle cannot be null"));
    }

    @Override
    protected MemorySegment nullHandle() {
        return MemorySegment.NULL;
    }

    @Override
    protected boolean isNullHandle(MemorySegment h) {
        return h == null || h.equals(MemorySegment.NULL);
    }

    @Override
    protected MemorySegment acquireReadLease() {
        return super.acquireReadLease();
    }

    @Override
    protected void releaseReadLease() {
        super.releaseReadLease();
    }

    @Override
    protected abstract String resourceName();

    /**
     * Scoped RAII token guaranteeing that the underlying native resource
     * remains leased and valid until {@link #close()} is called.
     */
    public final class Lease implements AutoCloseable {
        private final MemorySegment leasedHandle;
        private final Thread owner = Thread.currentThread();
        private boolean released = false;

        private Lease(MemorySegment leasedHandle) {
            this.leasedHandle = leasedHandle;
        }

        public MemorySegment handle() {
            if (released) {
                throw new IllegalStateException("Lease for " + resourceName() + " has already been closed");
            }
            return leasedHandle;
        }

        @Override
        public void close() {
            if (released) {
                return;
            }
            if (Thread.currentThread() != owner) {
                throw new IllegalStateException("Lease for " + resourceName() + " must be closed by its owning thread (" + owner.getName() + "), but was called by " + Thread.currentThread().getName());
            }
            releaseReadLease();
            released = true;
        }
    }

    public Lease lease() {
        return new Lease(acquireReadLease());
    }

    @FunctionalInterface
    public interface ResourceAction<R, E extends Throwable> extends HandleAction<MemorySegment, R, E> {
    }

    protected <R, E extends Throwable> R withHandle(ResourceAction<R, E> action) throws E {
        return super.withHandle(action);
    }

    @Deprecated(forRemoval = true)
    public MemorySegment getHandle() {
        return unsafeBorrowedHandle();
    }
}
