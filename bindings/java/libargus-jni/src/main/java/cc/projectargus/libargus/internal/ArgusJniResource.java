package cc.projectargus.libargus.internal;

import java.util.Objects;
import java.util.function.LongConsumer;
import java.util.function.LongFunction;

/**
 * Base class for JNI native resources backed by a 64-bit native pointer handle ({@code jlong}).
 * Extends {@link AbstractNativeResource} to provide thread-safe read-write handle leasing and
 * idempotent lifecycle destruction on JDK 17+.
 */
public abstract class ArgusJniResource extends AbstractNativeResource<Long> {

    protected ArgusJniResource(long handle) {
        super(validateNonZero(handle));
    }

    private static Long validateNonZero(long handle) {
        if (handle == 0L) {
            throw new IllegalArgumentException("Native handle cannot be 0 (NULL)");
        }
        return handle;
    }

    @Override
    protected final Long nullHandle() {
        return 0L;
    }

    @Override
    protected final boolean isNullHandle(Long handle) {
        return handle == null || handle == 0L;
    }

    @Override
    protected final void releaseNative(Long oldHandle) {
        if (oldHandle != null && oldHandle != 0L) {
            releaseNativeHandle(oldHandle);
        }
    }

    /**
     * Releases the underlying 64-bit native pointer.
     */
    protected abstract void releaseNativeHandle(long oldHandle);

    @Override
    protected final Long acquireReadLease() {
        return super.acquireReadLease();
    }

    @Override
    protected final void releaseReadLease() {
        super.releaseReadLease();
    }

    @Override
    protected abstract String resourceName();

    /**
     * Acquires a read lease and returns the primitive 64-bit native pointer.
     */
    protected final long acquireReadLeaseHandle() {
        return super.acquireReadLease();
    }

    /**
     * Executes a void operation with a safely leased primitive 64-bit native pointer.
     */
    protected final void withHandleVoid(LongConsumer action) {
        Objects.requireNonNull(action, "action");
        long h = acquireReadLease();
        try {
            action.accept(h);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the raw 64-bit native pointer without acquiring a lease.
     * For internal package diagnostics and identity comparison only.
     */
    public final long unsafeBorrowedHandleLong() {
        Long h = unsafeBorrowedHandle();
        return h != null ? h : 0L;
    }

    /**
     * Scoped read-lock lease for passing a 64-bit native handle across collaborating objects.
     */
    public final class Lease implements AutoCloseable {
        private final long leasedHandle;
        private boolean released = false;

        private Lease() {
            this.leasedHandle = acquireReadLease();
        }

        public long handle() {
            if (released) {
                throw new IllegalStateException("Lease has already been closed for " + resourceName());
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
     * Acquires a scoped read-lock lease on this native resource.
     */
    public final Lease lease() {
        return new Lease();
    }
}
