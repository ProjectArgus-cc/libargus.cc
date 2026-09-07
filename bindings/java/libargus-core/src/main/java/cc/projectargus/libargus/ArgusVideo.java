package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Iterates over video stream frames using unmanaged FFmpeg decoders.
 * Extends {@link ArgusNativeResource} for safe native leasing and idempotent lifecycle deallocation.
 */
public final class ArgusVideo extends ArgusNativeResource {

    private ArgusVideo(MemorySegment videoPtr) {
        super(videoPtr);
    }

    @Override
    protected String resourceName() {
        return "ArgusVideo";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_video_free.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        }
    }

    /**
     * Prepares a video processing session from a file.
     */
    public static ArgusVideo loadFile(Arena arena, ArgusMultimodalContext mctx, Path filePath, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(filePath);

        MemorySegment mctxH = mctx.acquireReadLease();
        try {
            MemorySegment pathSeg = arena.allocateFrom(filePath.toAbsolutePath().toString());
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_video_load_file.invokeExact(
                mctxH,
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
        } finally {
            mctx.releaseReadLease();
        }
    }

    /**
     * Prepares a video processing session from an in-memory buffer.
     */
    public static ArgusVideo loadBuffer(Arena arena, ArgusMultimodalContext mctx, byte[] buffer, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(buffer);

        MemorySegment mctxH = mctx.acquireReadLease();
        try {
            MemorySegment bufferSeg = arena.allocateFrom(ValueLayout.JAVA_BYTE, buffer);
            MemorySegment ptr = (MemorySegment) ArgusBindings.argus_video_load_buffer.invokeExact(
                mctxH,
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
        } finally {
            mctx.releaseReadLease();
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
        MemorySegment videoH = acquireReadLease();
        try (Arena local = Arena.ofConfined()) {
            MemorySegment outBitmapSeg = local.allocate(ValueLayout.ADDRESS);
            MemorySegment outTextSeg = local.allocate(256);

            int res = (int) ArgusBindings.argus_video_read_next.invokeExact(videoH, outBitmapSeg, outTextSeg, 256);
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
        } finally {
            releaseReadLease();
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
        MemorySegment videoH = acquireReadLease();
        try {
            MemorySegment outBitmapSeg = item.outBitmapSeg();
            MemorySegment outTextSeg = item.outTextSeg();
            outBitmapSeg.set(ValueLayout.ADDRESS, 0, MemorySegment.NULL);
            outTextSeg.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);

            int res = (int) ArgusBindings.argus_video_read_next.invokeExact(videoH, outBitmapSeg, outTextSeg, 256);
            if (res == 0) {
                MemorySegment bitmapHandle = outBitmapSeg.get(ValueLayout.ADDRESS, 0);
                String text = outTextSeg.getString(0);
                if (text.isEmpty()) {
                    text = null;
                }
                item.update(bitmapHandle, text);
                return true;
            } else if (res == -1) {
                item.update(null, null);
                return false; // EOF
            } else {
                ArgusNativeException.checkStatus(res, "argus_video_read_next");
                return false;
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to read next item from video stream", t);
        } finally {
            releaseReadLease();
        }
    }
}
