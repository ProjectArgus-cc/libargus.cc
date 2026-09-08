package cc.projectargus.libargus.internal;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Deterministic test fixture subclassing {@link ArgusNativeResource}.
 * Enables verifying handle leasing, thread affinity, write-lock exclusion,
 * and double-close idempotence without invoking native C++ library destructors.
 */
public final class TestNativeResource extends ArgusNativeResource {
    private final AtomicInteger releaseCount = new AtomicInteger(0);
    private final AtomicReference<MemorySegment> releasedHandle = new AtomicReference<>(null);

    private final CountDownLatch enterLatch = new CountDownLatch(1);
    private final CountDownLatch releaseLatch = new CountDownLatch(1);

    public TestNativeResource(MemorySegment handle) {
        super(handle);
    }

    @Override
    protected String resourceName() {
        return "TestNativeResource";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        releaseCount.incrementAndGet();
        releasedHandle.set(oldHandle);
    }

    public int getReleaseCount() {
        return releaseCount.get();
    }

    public MemorySegment getReleasedHandle() {
        return releasedHandle.get();
    }

    public CountDownLatch getEnterLatch() {
        return enterLatch;
    }

    public CountDownLatch getReleaseLatch() {
        return releaseLatch;
    }

    /**
     * Executes a real public operation under the read lease, pausing at a deterministic
     * synchronization point until {@link #getReleaseLatch()} is released.
     */
    public void executeBlockingOperation() {
        withHandle(h -> {
            enterLatch.countDown();
            try {
                if (!releaseLatch.await(5, TimeUnit.SECONDS)) {
                    throw new RuntimeException("Timed out waiting for test latch release");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted during test blocking operation", e);
            }
            return null;
        });
    }

    /**
     * Executes an arbitrary action under the protected read lease.
     */
    public <R> R executeUnderLease(ResourceAction<R, Throwable> action) throws Throwable {
        return withHandle(action);
    }
}
