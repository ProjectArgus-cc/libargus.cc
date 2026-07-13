package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wraps raw/parsed media content in native memory.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusBitmap implements AutoCloseable {
    private MemorySegment bitmapPtr;

    ArgusBitmap(MemorySegment bitmapPtr) {
        this.bitmapPtr = Objects.requireNonNull(bitmapPtr);
    }

    void setHandle(MemorySegment handle) {
        this.bitmapPtr = Objects.requireNonNull(handle);
    }

    /**
     * Creates a bitmap from a raw RGB image segment (RGBRGB... format).
     */
    public static ArgusBitmap fromRgb(int width, int height, MemorySegment rgbDataSeg) {
        Objects.requireNonNull(rgbDataSeg);
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_from_rgb.invokeExact(width, height, rgbDataSeg);
            if (ptr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_bitmap_from_rgb returned NULL");
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to create ArgusBitmap from RGB data", t);
        }
    }

    /**
     * Creates a bitmap from a raw float PCM audio segment (e.g. 16kHz).
     */
    public static ArgusBitmap fromPcm(MemorySegment pcmDataSeg, int nSamples) {
        Objects.requireNonNull(pcmDataSeg);
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_bitmap_from_pcm.invokeExact(pcmDataSeg, nSamples);
            if (ptr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_bitmap_from_pcm returned NULL");
            }
            return new ArgusBitmap(ptr);
        } catch (Throwable t) {
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
            throw new RuntimeException("Failed to load ArgusBitmap from memory buffer", t);
        }
    }

    public MemorySegment getHandle() {
        return bitmapPtr;
    }

    @Override
    public synchronized void close() {
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
