package cc.projectargus.libargus;

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Reusable, mutable carrier yielding items from a video stream via the JNI bridge.
 */
public final class ArgusVideoItem implements AutoCloseable {
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private final long[] outBitmapHandle = new long[1];
    private final byte[][] outTextBytes = new byte[1][];
    private ArgusBitmap bitmap;
    private String text;
    private volatile boolean closed = false;

    long[] outBitmapHandle() {
        return outBitmapHandle;
    }

    byte[][] outTextBytes() {
        return outTextBytes;
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

    public boolean isClosed() {
        return closed;
    }

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

    void update(long newBitmapHandle, String newText) {
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("ArgusVideoItem is closed");
            }
            if (bitmap != null) {
                if (bitmap.isClosed() || bitmap.unsafeBorrowedHandleLong() != newBitmapHandle) {
                    bitmap.close();
                    bitmap = null;
                }
            }
            if (newBitmapHandle != 0L && bitmap == null) {
                bitmap = new ArgusBitmap(newBitmapHandle);
            }
            this.text = newText;
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    public void setMergeable(boolean mergeable) {
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("ArgusVideoItem is closed");
            }
            if (bitmap != null) {
                bitmap.setMergeable(mergeable);
            }
        } finally {
            lifecycleLock.readLock().unlock();
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
            outBitmapHandle[0] = 0L;
            outTextBytes[0] = null;
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
