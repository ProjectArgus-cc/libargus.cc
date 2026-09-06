package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Iterates over video stream frames using unmanaged FFmpeg decoders.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusVideo implements AutoCloseable {
    private MemorySegment videoPtr;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private ArgusVideo(MemorySegment videoPtr) {
        this.videoPtr = Objects.requireNonNull(videoPtr);
    }

    /**
     * Prepares a video processing session from a file.
     */
    public static ArgusVideo loadFile(Arena arena, ArgusMultimodalContext mctx, Path filePath, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(filePath);

        MemorySegment pathSeg = arena.allocateFrom(filePath.toAbsolutePath().toString());
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_video_load_file.invokeExact(
                mctx.getHandle(),
                pathSeg,
                fpsTarget,
                timestampIntervalMs
            );
            if (ptr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_video_load_file");
            }
            return new ArgusVideo(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load ArgusVideo from file: " + filePath, t);
        }
    }

    /**
     * Prepares a video processing session from an in-memory buffer.
     */
    public static ArgusVideo loadBuffer(Arena arena, ArgusMultimodalContext mctx, byte[] buffer, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(buffer);

        MemorySegment bufferSeg = arena.allocateFrom(ValueLayout.JAVA_BYTE, buffer);
        try {
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_video_load_buffer.invokeExact(
                mctx.getHandle(),
                bufferSeg,
                buffer.length,
                fpsTarget,
                timestampIntervalMs
            );
            if (ptr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_video_load_buffer");
            }
            return new ArgusVideo(ptr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load ArgusVideo from memory buffer", t);
        }
    }

    /**
     * Represents a single item yielded from the video stream.
     * Contains either a newly decoded frame bitmap OR a timestamp text snippet.
     */
    public record VideoItem(ArgusBitmap bitmap, String text) {}

    /**
     * Reads the next frame bitmap or text timestamp from the video stream.
     *
     * @return a VideoItem, or null if EOF is reached
     * @deprecated Use {@link #readNext(ArgusVideoItem)} to prevent JVM allocation churn on hot paths.
     */
    @Deprecated
    public VideoItem readNext() {
        checkNotClosed();
        try (Arena local = Arena.ofConfined()) {
            MemorySegment outBitmapSeg = local.allocate(ValueLayout.ADDRESS);
            MemorySegment outTextSeg = local.allocate(256);

            int res = (int) ArgusBindings.argus_video_read_next.invokeExact(videoPtr, outBitmapSeg, outTextSeg, 256);
            if (res == 0) {
                MemorySegment bitmapHandle = outBitmapSeg.get(ValueLayout.ADDRESS, 0);
                String text = outTextSeg.getString(0);
                if (text.isEmpty()) {
                    text = null;
                }

                ArgusBitmap bitmap = null;
                if (!bitmapHandle.equals(MemorySegment.NULL)) {
                    bitmap = new ArgusBitmap(bitmapHandle);
                }

                return new VideoItem(bitmap, text);
            } else if (res == -1) {
                return null; // EOF
            } else {
                ArgusNativeException.checkStatus(res, "argus_video_read_next");
                return null;
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to read next item from video stream", t);
        }
    }

    /**
     * Reads the next frame bitmap or text timestamp from the video stream into the provided mutable carrier.
     * Reuses off-heap resources and prevents JVM heap allocation.
     *
     * @param item the mutable carrier container to update
     * @return true if an item was successfully read, false if EOF is reached
     */
    public boolean readNext(ArgusVideoItem item) {
        Objects.requireNonNull(item);
        checkNotClosed();
        try (Arena local = Arena.ofConfined()) {
            MemorySegment outBitmapSeg = local.allocate(ValueLayout.ADDRESS);
            MemorySegment outTextSeg = local.allocate(256);

            int res = (int) ArgusBindings.argus_video_read_next.invokeExact(videoPtr, outBitmapSeg, outTextSeg, 256);
            if (res == 0) {
                MemorySegment bitmapHandle = outBitmapSeg.get(ValueLayout.ADDRESS, 0);
                String text = outTextSeg.getString(0);
                if (text.isEmpty()) {
                    text = null;
                }
                item.update(bitmapHandle, text);
                return true;
            } else if (res == -1) {
                return false; // EOF
            } else {
                ArgusNativeException.checkStatus(res, "argus_video_read_next");
                return false;
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to read next item from video stream", t);
        }
    }

    public MemorySegment getHandle() {
        checkNotClosed();
        return videoPtr;
    }

    public boolean isClosed() {
        return closed.get();
    }

    private void checkNotClosed() {
        if (closed.get() || videoPtr == null || videoPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("ArgusVideo session has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (videoPtr != null && !videoPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_video_free.invokeExact(videoPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native video resources", t);
            } finally {
                videoPtr = MemorySegment.NULL;
            }
        }
    }
}
