package cc.projectargus.libargus.internal;

import java.nio.Buffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Spatial boundary, arithmetic overflow, and direct-buffer validation utilities for {@code :libargus-jni}.
 */
public final class ArgusJniValidation {

    private ArgusJniValidation() {}

    public static void checkPositive(long value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0 (got " + value + ")");
        }
    }

    public static void checkNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 (got " + value + ")");
        }
    }

    public static long multiplyExactBytes(long count, long elementBytes, String name) {
        checkNonNegative(count, name + " count");
        checkPositive(elementBytes, name + " elementBytes");
        try {
            return Math.multiplyExact(count, elementBytes);
        } catch (ArithmeticException ex) {
            throw new IllegalArgumentException("Byte size overflow for " + name + ": " + count + " * " + elementBytes, ex);
        }
    }

    public static void checkDirectBuffer(Buffer buffer, long requiredElements, String name) {
        Objects.requireNonNull(buffer, name);
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException(name + " must be a direct NIO buffer (allocateDirect)");
        }
        if (buffer.remaining() < requiredElements) {
            throw new IllegalArgumentException(
                name + " remaining elements (" + buffer.remaining() + ") < required (" + requiredElements + ")"
            );
        }
    }

    public static void checkDirectWritableBuffer(Buffer buffer, long requiredElements, String name) {
        checkDirectBuffer(buffer, requiredElements, name);
        if (buffer.isReadOnly()) {
            throw new IllegalArgumentException(name + " is read-only; writable direct buffer required");
        }
    }

    /**
     * Encodes a Java String to Standard UTF-8 bytes, rejecting embedded NUL characters when used as a C path/key.
     */
    public static byte[] toUtf8NoNull(String str, String name) {
        Objects.requireNonNull(str, name);
        if (str.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(name + " must not contain embedded NUL characters");
        }
        return str.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Encodes a Java String to Standard UTF-8 bytes (allowing arbitrary content for length-explicit APIs).
     */
    public static byte[] toUtf8(String str, String name) {
        Objects.requireNonNull(str, name);
        return str.getBytes(StandardCharsets.UTF_8);
    }
}
