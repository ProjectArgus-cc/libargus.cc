package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Wraps raw/parsed media content in native memory.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusBitmap implements AutoCloseable {
    private MemorySegment bitmapPtr;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    ArgusBitmap(MemorySegment bitmapPtr) {
        this.bitmapPtr = Objects.requireNonNull(bitmapPtr);
    }

    void setHandle(MemorySegment handle) {
        checkNotClosed();
        this.bitmapPtr = Objects.requireNonNull(handle);
    }

    /**
     * Creates a bitmap from a raw RGB image segment (RGBRGB... format).
     */
    public static ArgusBitmap fromRgb(int width, int height, MemorySegment rgbDataSeg) {
        Objects.requireNonNull(rgbDataSeg);
        ArgusValidation.checkPositive(width, "width");
        ArgusValidation.checkPositive(height, "height");
        long numPixels = ArgusValidation.multiplyExactBytes(width, height, "pixels");
        long requiredBytes = ArgusValidation.multiplyExactBytes(numPixels, 3, "rgbDataSeg");
        ArgusValidation.checkReadable(rgbDataSeg, requiredBytes, "rgbDataSeg");

        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_from_rgb.invokeExact(width, height, rgbDataSeg);
            if (ptr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_bitmap_from_rgb");
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to create ArgusBitmap from RGB data", t);
        }
    }

    /**
     * Creates a bitmap from a raw float PCM audio segment (e.g. 16kHz).
     */
    public static ArgusBitmap fromPcm(MemorySegment pcmDataSeg, int nSamples) {
        Objects.requireNonNull(pcmDataSeg);
        ArgusValidation.checkPositive(nSamples, "nSamples");
        long requiredBytes = ArgusValidation.multiplyExactBytes(nSamples, ValueLayout.JAVA_FLOAT.byteSize(), "pcmDataSeg");
        ArgusValidation.checkReadable(pcmDataSeg, requiredBytes, "pcmDataSeg");

        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_from_pcm.invokeExact(pcmDataSeg, nSamples);
            if (ptr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_bitmap_from_pcm");
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to create ArgusBitmap from PCM data", t);
        }
    }

    /**
     * Automatically loads and processes media from a local file path.
     */
    public static ArgusBitmap loadFile(Arena arena, ArgusMultimodalContext mctx, Path filePath, boolean placeholder) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(filePath);

        MemorySegment pathSeg = arena.allocateFrom(filePath.toAbsolutePath().toString());
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_load_file.invokeExact(
                mctx.getHandle(),
                pathSeg,
                placeholder
            );
            if (ptr.equals(MemorySegment.NULL)) {
                return null;
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load ArgusBitmap from file: " + filePath, t);
        }
    }

    /**
     * Automatically loads and processes media from a memory buffer.
     */
    public static ArgusBitmap loadBuffer(Arena arena, ArgusMultimodalContext mctx, byte[] buffer, boolean placeholder) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(buffer);

        MemorySegment bufferSeg = arena.allocateFrom(ValueLayout.JAVA_BYTE, buffer);
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_load_buffer.invokeExact(
                mctx.getHandle(),
                bufferSeg,
                buffer.length,
                placeholder
            );
            if (ptr.equals(MemorySegment.NULL)) {
                return null;
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load ArgusBitmap from memory buffer", t);
        }
    }

    public MemorySegment getHandle() {
        checkNotClosed();
        return bitmapPtr;
    }

    public boolean isClosed() {
        return closed.get();
    }

    private void checkNotClosed() {
        if (closed.get() || bitmapPtr == null || bitmapPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("ArgusBitmap has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (bitmapPtr != null && !bitmapPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_bitmap_free.invokeExact(bitmapPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native bitmap resources", t);
            } finally {
                bitmapPtr = MemorySegment.NULL;
            }
        }
    }
}
