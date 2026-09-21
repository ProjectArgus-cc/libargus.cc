package cc.projectargus.libargus;

/**
 * Diagnostic log level matching ggml severity tiers.
 */
public enum ArgusLogLevel {
    /** Completely silent; no log emissions. */
    NONE(0),
    /** Fine-grained diagnostic and debug telemetry. */
    DEBUG(1),
    /** Informational status (e.g. model geometry, timings). */
    INFO(2),
    /** Warning conditions. */
    WARN(3),
    /** Critical errors and execution failures. */
    ERROR(4),
    /** Continuation chunk of previous log line. */
    CONT(5);

    private final int value;

    ArgusLogLevel(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static ArgusLogLevel fromValue(int value) {
        for (ArgusLogLevel level : values()) {
            if (level.value == value) {
                return level;
            }
        }
        return NONE;
    }
}
