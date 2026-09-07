package cc.projectargus.libargus;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Reusable, mutable carrier yielding items from a video stream.
 * Prevents JVM heap allocation churn on the hot iteration path.
 */
public final class ArgusVideoItem implements AutoCloseable {
    private final Arena arena = Arena.ofShared();
    private final MemorySegment outBitmapSeg = arena.allocate(ValueLayout.ADDRESS);
    private final MemorySegment outTextSeg = arena.allocate(256);
    private ArgusBitmap bitmap;
    private String text;

    MemorySegment outBitmapSeg() {
        return outBitmapSeg;
    }

    MemorySegment outTextSeg() {
        return outTextSeg;
    }

    /**
     * Returns the frame bitmap, or null if this item contains timestamp text.
     */
    public ArgusBitmap bitmap() {
        return bitmap;
    }

    /**
     * Returns the timestamp text snippet, or null if this item contains a frame bitmap.
     */
    public String text() {
        return text;
    }

    /**
     * Updates the container state in-place.
     * Automatically frees the previous native bitmap resources to prevent memory leaks.
     */
    void update(MemorySegment newBitmapPtr, String newText) {
        MemorySegment safeBitmapPtr = (newBitmapPtr != null) ? newBitmapPtr : MemorySegment.NULL;
        if (bitmap != null) {
            if (bitmap.isClosed() || !bitmap.getHandle().equals(safeBitmapPtr)) {
                bitmap.close();
                bitmap = null;
            }
        }
        if (!safeBitmapPtr.equals(MemorySegment.NULL) && bitmap == null) {
            bitmap = new ArgusBitmap(safeBitmapPtr);
        }
        this.text = newText;
    }

    @Override
    public synchronized void close() {
        if (bitmap != null) {
            bitmap.close();
            bitmap = null;
        }
        text = null;
        arena.close();
    }
}
