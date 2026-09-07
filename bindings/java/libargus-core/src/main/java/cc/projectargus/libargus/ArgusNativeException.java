package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Exception thrown when a native libargus C ABI downcall returns an error status
 * or sets a thread-local diagnostic error code.
 */
public class ArgusNativeException extends RuntimeException {
    private final int errorCode;

    public ArgusNativeException(int errorCode, String message) {
        super(String.format("[Native Error %d] %s", errorCode, message));
        this.errorCode = errorCode;
    }

    public ArgusNativeException(int errorCode, String message, Throwable cause) {
        super(String.format("[Native Error %d] %s", errorCode, message), cause);
        this.errorCode = errorCode;
    }

    /**
     * Retrieves the native error code (matching argus_error_code_t).
     */
    public int getErrorCode() {
        return errorCode;
    }

    /**
     * Checks a native return status and throws ArgusNativeException if negative.
     *
     * @param status native function return code
     * @param operation diagnostic operation label
     */
    public static void checkStatus(int status, String operation) {
        if (status < 0) {
            int code = 8; // ARGUS_ERROR_INTERNAL fallback
            String msg = "Unknown native failure";
            try {
                code = (int) ArgusBindings.argus_last_error_code.invokeExact();
                try (Arena localArena = Arena.ofConfined()) {
                    MemorySegment buf = localArena.allocate(512);
                    int len = (int) ArgusBindings.argus_last_error_message_copy.invokeExact(buf, 512);
                    if (len > 0) {
                        msg = buf.getString(0);
                    }
                }
            } catch (Throwable t) {
                // Ignore downcall introspection errors
            } finally {
                try {
                    ArgusBindings.argus_clear_error.invokeExact();
                } catch (Throwable ignored) {
                }
            }
            throw new ArgusNativeException(code, operation + " failed with status " + status + ": " + msg);
        }
    }
}
