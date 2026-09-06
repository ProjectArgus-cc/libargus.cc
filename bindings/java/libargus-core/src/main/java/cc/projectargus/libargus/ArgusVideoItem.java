package cc.projectargus.libargus;

import java.lang.foreign.MemorySegment;

/**
 * Reusable, mutable carrier yielding items from a video stream.
 * Prevents JVM heap allocation churn on the hot iteration path.
 */
public final class ArgusVideoItem implements AutoCloseable {
    private ArgusBitmap bitmap;
    private String text;

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
            if (!bitmap.getHandle().equals(safeBitmapPtr)) {
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
    }
}
