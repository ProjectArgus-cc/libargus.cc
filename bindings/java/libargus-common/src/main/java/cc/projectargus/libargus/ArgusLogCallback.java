package cc.projectargus.libargus;

/**
 * Functional callback interface for receiving redirected native diagnostic log messages.
 */
@FunctionalInterface
public interface ArgusLogCallback {
    /**
     * Invoked when the native library emits a diagnostic log message.
     *
     * @param level Severity level of the log message
     * @param message Formatted log string fragment
     */
    void onLog(ArgusLogLevel level, String message);
}
