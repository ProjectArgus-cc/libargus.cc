package cc.projectargus.libargus.internal;

import java.util.concurrent.TimeUnit;

/** Test-only observation of the actual production lifecycle lock queue. */
public final class LifecycleProbe {
    private LifecycleProbe() {}
    public static void awaitQueued(ArgusNativeResource resource, Thread thread) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!resource.lifecycleLock.hasQueuedThread(thread)) {
            if (!thread.isAlive() || System.nanoTime() >= deadline)
                throw new AssertionError("Thread never queued on the resource lifecycle lock");
            Thread.onSpinWait();
        }
    }
}
