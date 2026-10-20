package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;

/**
 * A thread-safe native atomic cancellation object ({@code argus_abort_flag_t})
 * to signal early termination to long-running native prefill and decode loops.
 * Extends {@link ArgusJniResource} for safe native leasing and idempotent lifecycle deallocation.
 */
public final class ArgusAbortFlag extends ArgusJniResource {

    public ArgusAbortFlag() {
        super(createNativeHandle());
    }

    private static long createNativeHandle() {
        long h = ArgusJni.nativeAbortFlagCreate();
        if (h == 0L) {
            throw new ArgusNativeException(ArgusNativeException.ERROR_OUT_OF_MEMORY, "Failed to allocate native argus_abort_flag_t");
        }
        return h;
    }

    @Override
    protected String resourceName() {
        return "ArgusAbortFlag";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeAbortFlagRelease(oldHandle);
    }

    /**
     * Signals cancellation to any active or subsequent native execution loops.
     */
    public void abort() {
        withHandleVoid(ArgusJni::nativeAbortFlagRequest);
    }

    /**
     * Resets the cancellation flag back to false.
     */
    public void reset() {
        withHandleVoid(ArgusJni::nativeAbortFlagReset);
    }

    /**
     * Checks if cancellation has been requested.
     */
    public boolean isAborted() {
        return withHandle(ArgusJni::nativeAbortFlagIsRequested);
    }
}
