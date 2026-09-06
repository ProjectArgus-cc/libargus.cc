package cc.projectargus.libargus.internal;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Spatial boundary, overflow safety, and segment validity assertions for Project Panama downcalls.
 */
public final class ArgusValidation {
    private ArgusValidation() {}

    /**
     * Asserts that a segment is non-null, writable, and has at least requiredBytes available.
     *
     * @param segment target MemorySegment
     * @param requiredBytes minimum required byte capacity
     * @param paramName parameter label for diagnostics
     * @throws NullPointerException if segment is null
     * @throws IllegalArgumentException if segment is read-only or smaller than requiredBytes
     */
    public static void checkWritable(MemorySegment segment, long requiredBytes, String paramName) {
        Objects.requireNonNull(segment, paramName + " must not be null");
        if (segment.isReadOnly()) {
            throw new IllegalArgumentException(paramName + " must be writable (provided segment is read-only)");
        }
        if (segment.byteSize() < requiredBytes) {
            throw new IllegalArgumentException(String.format(
                "%s capacity is insufficient: requires at least %d bytes, but segment has %d bytes",
                paramName, requiredBytes, segment.byteSize()
            ));
        }
    }

    /**
     * Asserts that a segment is non-null and has at least requiredBytes available.
     *
     * @param segment target MemorySegment
     * @param requiredBytes minimum required byte capacity
     * @param paramName parameter label for diagnostics
     * @throws NullPointerException if segment is null
     * @throws IllegalArgumentException if segment is smaller than requiredBytes
     */
    public static void checkReadable(MemorySegment segment, long requiredBytes, String paramName) {
        Objects.requireNonNull(segment, paramName + " must not be null");
        if (segment.byteSize() < requiredBytes) {
            throw new IllegalArgumentException(String.format(
                "%s capacity is insufficient: requires at least %d bytes, but segment has %d bytes",
                paramName, requiredBytes, segment.byteSize()
            ));
        }
    }

    /**
     * Multiplies count and element size using overflow-safe arithmetic.
     *
     * @param count number of elements
     * @param elementSize size of each element in bytes
     * @param paramName parameter label for diagnostics
     * @return total bytes required
     * @throws IllegalArgumentException if count is negative or multiplication overflows
     */
    public static long multiplyExactBytes(long count, long elementSize, String paramName) {
        if (count < 0) {
            throw new IllegalArgumentException(paramName + " count must be non-negative: " + count);
        }
        try {
            return Math.multiplyExact(count, elementSize);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Byte size calculation overflow for " + paramName + ": " + count + " * " + elementSize, e);
        }
    }

    /**
     * Asserts that a value is positive (> 0).
     */
    public static void checkPositive(long value, String paramName) {
        if (value <= 0) {
            throw new IllegalArgumentException(paramName + " must be strictly positive: " + value);
        }
    }

    /**
     * Asserts that a value is non-negative (>= 0).
     */
    public static void checkNonNegative(long value, String paramName) {
        if (value < 0) {
            throw new IllegalArgumentException(paramName + " must be non-negative: " + value);
        }
    }
}
