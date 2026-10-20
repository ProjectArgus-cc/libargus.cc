package cc.projectargus.libargus.internal;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Spatial boundary, overflow safety, and segment validity assertions for Project Panama downcalls.
 */
public final class ArgusValidation {
    private ArgusValidation() {}

    public static void checkAlignment(MemorySegment segment, long alignment, String name) {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0 ||
                (segment.address() & (alignment - 1)) != 0) {
            throw new IllegalArgumentException(name + " is not aligned to " + alignment);
        }
    }


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
        Objects.requireNonNull(segment);
        if (!segment.isNative()) throw new IllegalArgumentException(paramName + " must be native memory");
        if (!segment.scope().isAlive()) throw new IllegalStateException(paramName + " scope is closed");
        if (!segment.isAccessibleBy(Thread.currentThread())) throw new WrongThreadException(paramName);

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
        Objects.requireNonNull(segment);
        if (!segment.isNative()) throw new IllegalArgumentException(paramName + " must be native memory");
        if (!segment.scope().isAlive()) throw new IllegalStateException(paramName + " scope is closed");
        if (!segment.isAccessibleBy(Thread.currentThread())) throw new WrongThreadException(paramName);

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
        return ArgusCommonValidation.multiplyExactBytes(count, elementSize, paramName);
    }

    /**
     * Asserts that a value is positive (> 0).
     */
    public static void checkPositive(long value, String paramName) {
        ArgusCommonValidation.checkPositive(value, paramName);
    }

    /**
     * Asserts that a value is non-negative (>= 0).
     */
    public static void checkNonNegative(long value, String paramName) {
        ArgusCommonValidation.checkNonNegative(value, paramName);
    }

    /**
     * Asserts that a value fits within a signed 32-bit integer range [0, Integer.MAX_VALUE].
     */
    public static int checkIntBounds(long value, String paramName) {
        return ArgusCommonValidation.checkIntBounds(value, paramName);
    }

    /**
     * Asserts that a segment has positive length and contains a NUL terminator within its bounds.
     */
    public static void checkNullTerminated(MemorySegment segment, String paramName) {
        Objects.requireNonNull(segment);
        if (!segment.isNative()) throw new IllegalArgumentException(paramName + " must be native memory");
        if (!segment.scope().isAlive()) throw new IllegalStateException(paramName + " scope is closed");
        if (!segment.isAccessibleBy(Thread.currentThread())) throw new WrongThreadException(paramName);

        long size = segment.byteSize();
        if (size <= 0) {
            throw new IllegalArgumentException(paramName + " must have positive capacity and be null-terminated");
        }
        boolean found = false;
        for (long i = 0; i < size; i++) {
            if (segment.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw new IllegalArgumentException(paramName + " does not contain null terminator within its bounds");
        }
    }

    /**
     * Allocates an off-heap UTF-8 memory segment for a Java string without appending
     * an artificial C null terminator byte.
     *
     * @param arena allocator arena
     * @param text string to convert
     * @return memory segment whose byteSize() exactly equals the UTF-8 byte length
     */
    public static MemorySegment allocateUtf8Text(java.lang.foreign.Arena arena, String text) {
        Objects.requireNonNull(arena, "arena");
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) {
            return arena.allocate(0);
        }
        MemorySegment seg = arena.allocateFrom(text);
        return seg.asSlice(0, seg.byteSize() - 1);
    }
}
