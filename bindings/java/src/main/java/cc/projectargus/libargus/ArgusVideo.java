package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Iterates over video stream frames using unmanaged FFmpeg decoders.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusVideo implements AutoCloseable {
    private MemorySegment videoPtr;

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
                throw new RuntimeException("Native argus_video_load_file returned NULL for: " + filePath);
            }
            return new ArgusVideo(ptr);
        } catch (Throwable t) {
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
                throw new RuntimeException("Native argus_video_load_buffer returned NULL");
            }
            return new ArgusVideo(ptr);
        } catch (Throwable t) {
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
     */
    public VideoItem readNext() {
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
                    // We must wrap the native pointer in our custom wrapper class.
                    // Construct a new instance manually. We can do a private/package-private constructor mapping.
                    // To do this, we use a custom factory method, but since the constructor is package-private here,
                    // we can just construct it.
                    bitmap = new ArgusBitmap(bitmapHandle);
                }

                return new VideoItem(bitmap, text);
            } else if (res == -1) {
                return null; // EOF
            } else {
                throw new RuntimeException("Native argus_video_read_next failed with code: " + res);
            }
        } catch (Throwable t) {
            throw new RuntimeException("Failed to read next item from video stream", t);
        }
    }

    public MemorySegment getHandle() {
        return videoPtr;
    }

    @Override
    public synchronized void close() {
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
