package cc.projectargus.libargus;

/**
 * Exception thrown when a native libargus C ABI downcall or JNI call returns an error status
 * or sets a thread-local diagnostic error code.
 */
public class ArgusNativeException extends RuntimeException {
    public static final int ERROR_SUCCESS = 0;
    public static final int ERROR_INVALID_ARGUMENT = 1;
    public static final int ERROR_OUT_OF_MEMORY = 2;
    public static final int ERROR_BACKEND = 3;
    public static final int ERROR_MODEL_LOAD = 4;
    public static final int ERROR_DECODE = 5;
    public static final int ERROR_CANCELLED = 6;
    public static final int ERROR_BUSY = 7;
    public static final int ERROR_INTERNAL = 8;
    public static final int ERROR_ROLLBACK_FAILED = 9;
    public static final int ERROR_INVALID_STATE = 10;

    /**
     * Pluggable error resolver registered by the active binding transport (Panama FFM or JNI).
     */
    public interface ErrorResolver {
        int lastErrorCode();
        String lastErrorMessage();
        void clearError();
    }

    private static volatile ErrorResolver errorResolver;

    public static void registerErrorResolver(ErrorResolver resolver) {
        if (resolver != null) {
            errorResolver = resolver;
        }
    }

    private static ErrorResolver resolveBridge() {
        ErrorResolver r = errorResolver;
        if (r != null) {
            return r;
        }
        for (String candidate : new String[]{
            "cc.projectargus.libargus.internal.ArgusBindings",
            "cc.projectargus.libargus.internal.ArgusJni"
        }) {
            try {
                Class.forName(candidate);
                r = errorResolver;
                if (r != null) {
                    return r;
                }
            } catch (Throwable ignored) {
            }
        }
        return null;
    }

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
            int code = ERROR_INTERNAL;
            String msg = "Unknown native failure";
            ErrorResolver resolver = resolveBridge();
            if (resolver != null) {
                try {
                    int c = resolver.lastErrorCode();
                    if (c != 0) {
                        code = c;
                    }
                    String m = resolver.lastErrorMessage();
                    if (m != null && !m.isEmpty()) {
                        msg = m;
                    }
                } catch (Throwable ignored) {
                } finally {
                    try {
                        resolver.clearError();
                    } catch (Throwable ignored) {
                    }
                }
            }
            throw new ArgusNativeException(code, operation + " failed with status " + status + ": " + msg);
        }
    }

    /**
     * Throws an ArgusNativeException populated with the thread-local native error code and message,
     * and clears the native error state.
     *
     * @param operation diagnostic operation label
     */
    public static void throwLastError(String operation) {
        int code = ERROR_INTERNAL;
        String msg = "Unknown native failure";
        ErrorResolver resolver = resolveBridge();
        if (resolver != null) {
            try {
                int c = resolver.lastErrorCode();
                if (c != 0) {
                    code = c;
                }
                String m = resolver.lastErrorMessage();
                if (m != null && !m.isEmpty()) {
                    msg = m;
                }
            } catch (Throwable ignored) {
            } finally {
                try {
                    resolver.clearError();
                } catch (Throwable ignored) {
                }
            }
        }
        throw new ArgusNativeException(code, operation + " failed: " + msg);
    }
}
