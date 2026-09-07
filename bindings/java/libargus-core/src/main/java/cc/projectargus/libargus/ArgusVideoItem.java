package cc.projectargus.libargus;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Reusable, mutable carrier yielding items from a video stream.
 * Reuses off-heap scratch buffers across iterations to avoid off-heap buffer reallocation.
 * Note: Each decoded frame may still construct an {@link ArgusBitmap} wrapper or JVM {@link String}.
 */
public final class ArgusVideoItem implements AutoCloseable {
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private final Arena arena = Arena.ofShared();
    private final MemorySegment outBitmapSeg = arena.allocate(ValueLayout.ADDRESS);
    private final MemorySegment outTextSeg = arena.allocate(256);
    private ArgusBitmap bitmap;
    private String text;
    private volatile boolean closed = false;

    MemorySegment outBitmapSeg() {
        return outBitmapSeg;
    }

    MemorySegment outTextSeg() {
        return outTextSeg;
    }

    void acquireWriteLock() {
        lifecycleLock.writeLock().lock();
        if (closed) {
            lifecycleLock.writeLock().unlock();
            throw new IllegalStateException("ArgusVideoItem is closed");
        }
    }

    void releaseWriteLock() {
        lifecycleLock.writeLock().unlock();
    }

    /**
     * Returns true if this item container has been closed.
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Returns the frame bitmap, or null if this item contains timestamp text.
     *
     * @throws IllegalStateException if this item container is closed
     */
    public ArgusBitmap bitmap() {
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("ArgusVideoItem is closed");
            }
            return bitmap;
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Returns the timestamp text snippet, or null if this item contains a frame bitmap.
     *
     * @throws IllegalStateException if this item container is closed
     */
    public String text() {
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("ArgusVideoItem is closed");
            }
            return text;
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Updates the container state in-place.
     * Automatically frees the previous native bitmap resources to prevent memory leaks.
     */
    void update(MemorySegment newBitmapPtr, String newText) {
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("ArgusVideoItem is closed");
            }
            MemorySegment safeBitmapPtr = (newBitmapPtr != null) ? newBitmapPtr : MemorySegment.NULL;
            if (bitmap != null) {
                if (bitmap.isClosed() || !bitmap.unsafeBorrowedHandle().equals(safeBitmapPtr)) {
                    bitmap.close();
                    bitmap = null;
                }
            }
            if (!safeBitmapPtr.equals(MemorySegment.NULL) && bitmap == null) {
                bitmap = new ArgusBitmap(safeBitmapPtr);
            }
            this.text = newText;
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            if (bitmap != null) {
                bitmap.close();
                bitmap = null;
            }
            text = null;
            arena.close();
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
