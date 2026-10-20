package cc.projectargus.libargus.internal;

import java.nio.Buffer;

/**
 * Spatial boundary, arithmetic overflow, and direct-buffer validation utilities for {@code :libargus-jni},
 * delegating directly to {@link ArgusCommonValidation}.
 */
public final class ArgusJniValidation {

    private ArgusJniValidation() {}

    public static void checkPositive(long value, String name) {
        ArgusCommonValidation.checkPositive(value, name);
    }

    public static void checkNonNegative(long value, String name) {
        ArgusCommonValidation.checkNonNegative(value, name);
    }

    public static long multiplyExactBytes(long count, long elementBytes, String name) {
        return ArgusCommonValidation.multiplyExactBytes(count, elementBytes, name);
    }

    public static void checkDirectBuffer(Buffer buffer, long requiredElements, String name) {
        ArgusCommonValidation.checkDirectBuffer(buffer, requiredElements, name);
    }

    public static void checkDirectWritableBuffer(Buffer buffer, long requiredElements, String name) {
        ArgusCommonValidation.checkDirectWritableBuffer(buffer, requiredElements, name);
    }

    public static byte[] toUtf8NoNull(String str, String name) {
        return ArgusCommonValidation.toUtf8NoNull(str, name);
    }

    public static byte[] toUtf8(String str, String name) {
        return ArgusCommonValidation.toUtf8(str, name);
    }
}
