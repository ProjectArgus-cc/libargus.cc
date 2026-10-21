package cc.projectargus.libargus.internal;

import java.nio.Buffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Shared JDK 17+ spatial boundary, arithmetic overflow, direct-buffer, and UTF-8 validation
 * primitives used across both {@code :libargus-core} (Panama FFM) and {@code :libargus-jni} (JNI).
 */
public final class ArgusCommonValidation {

    public static final int KV_TYPE_F16 = 0;
    public static final int KV_TYPE_Q8_0 = 8;
    public static final int KV_TYPE_Q4_0 = 2;
    public static final int KV_TYPE_Q4_1 = 3;

    private ArgusCommonValidation() {}

    /**
     * Asserts that a value is strictly positive ({@code > 0}).
     */
    public static void checkPositive(long value, String paramName) {
        if (value <= 0) {
            throw new IllegalArgumentException(paramName + " must be strictly positive: " + value);
        }
    }

    /**
     * Asserts that a value is non-negative ({@code >= 0}).
     */
    public static void checkNonNegative(long value, String paramName) {
        if (value < 0) {
            throw new IllegalArgumentException(paramName + " must be non-negative: " + value);
        }
    }

    /**
     * Asserts that a value fits within a signed 32-bit integer range {@code [0, Integer.MAX_VALUE]}.
     */
    public static int checkIntBounds(long value, String paramName) {
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(String.format(
                "%s exceeds 32-bit integer boundary: %d (max allowed: %d)",
                paramName, value, Integer.MAX_VALUE
            ));
        }
        return (int) value;
    }

    /**
     * Multiplies count and element size using overflow-safe arithmetic.
     */
    public static long multiplyExactBytes(long count, long elementSize, String paramName) {
        if (count < 0) {
            throw new IllegalArgumentException(paramName + " count must be non-negative: " + count);
        }
        if (elementSize <= 0) {
            throw new IllegalArgumentException(paramName + " elementSize must be strictly positive: " + elementSize);
        }
        try {
            return Math.multiplyExact(count, elementSize);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(
                "Byte size calculation overflow for " + paramName + ": " + count + " * " + elementSize, e
            );
        }
    }

    /**
     * Asserts that a NIO buffer is non-null, direct, and has at least {@code requiredElements} remaining.
     */
    public static void checkDirectBuffer(Buffer buffer, long requiredElements, String paramName) {
        Objects.requireNonNull(buffer, paramName);
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException(paramName + " must be a direct NIO buffer (allocateDirect)");
        }
        if (buffer.remaining() < requiredElements) {
            throw new IllegalArgumentException(
                paramName + " remaining elements (" + buffer.remaining() + ") < required (" + requiredElements + ")"
            );
        }
    }

    /**
     * Asserts that a NIO buffer is non-null, direct, writable, and has at least {@code requiredElements} remaining.
     */
    public static void checkDirectWritableBuffer(Buffer buffer, long requiredElements, String paramName) {
        checkDirectBuffer(buffer, requiredElements, paramName);
        if (buffer.isReadOnly()) {
            throw new IllegalArgumentException(paramName + " is read-only; writable direct buffer required");
        }
    }

    /**
     * Encodes a Java String to Standard UTF-8 bytes, rejecting embedded NUL characters when used as a C path/key.
     */
    public static byte[] toUtf8NoNull(String str, String paramName) {
        Objects.requireNonNull(str, paramName);
        if (str.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(paramName + " must not contain embedded NUL characters");
        }
        return str.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Encodes a Java String to Standard UTF-8 bytes (allowing arbitrary content for length-explicit APIs).
     */
    public static byte[] toUtf8(String str, String paramName) {
        Objects.requireNonNull(str, paramName);
        return str.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Asserts that {@code [offset, offset + count)} is a valid slice within an array of length {@code arrayLength}.
     */
    public static void checkArraySlice(int arrayLength, int offset, int count, String paramName) {
        checkNonNegative(offset, "offset");
        checkNonNegative(count, "count");
        if ((long) offset + (long) count > arrayLength) {
            throw new IllegalArgumentException(
                paramName + " slice [offset=" + offset + ", count=" + count + ") exceeds length (" + arrayLength + ")"
            );
        }
    }

    /**
     * Validates {@code seqMax} and {@code cloneSlots} invariants for context configurations.
     */
    public static void validateContextSlots(int seqMax, int cloneSlots) {
        checkNonNegative(seqMax, "seqMax");
        checkNonNegative(cloneSlots, "cloneSlots");
        if (cloneSlots > 0 && seqMax <= cloneSlots) {
            throw new IllegalArgumentException(
                "seqMax (" + seqMax + ") must exceed cloneSlots (" + cloneSlots + ") to reserve at least one base slot"
            );
        }
    }

    /**
     * Calculates the effective {@code seqMax} capacity incorporating reserved clone slots with overflow checking.
     */
    public static int computeEffectiveSeqMax(int seqMax, int cloneSlots, boolean requiresMultiSlotBase) {
        if (cloneSlots <= 0) {
            return seqMax;
        }
        int baseRequired = requiresMultiSlotBase ? 4 : 1;
        int base = seqMax > 0 ? seqMax : baseRequired;
        try {
            return Math.addExact(base, cloneSlots);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(
                "Integer overflow calculating effective seqMax (base=" + base + ", cloneSlots=" + cloneSlots + ")", e
            );
        }
    }

    /**
     * Returns the default CPU thread ceiling (half of available logical processors, minimum 1).
     */
    public static int defaultCpuThreads() {
        return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    }
}
