package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Iterates over video stream frames using unmanaged FFmpeg decoders via the JDK 17+ / Android JNI bridge.
 */
public final class ArgusVideo extends ArgusJniResource {

    private final ArgusMultimodalContext mctx;

    private ArgusVideo(long videoHandle, ArgusMultimodalContext mctx) {
        super(videoHandle);
        this.mctx = Objects.requireNonNull(mctx, "mctx cannot be null");
    }

    public ArgusMultimodalContext multimodalContext() {
        return mctx;
    }

    @Override
    protected String resourceName() {
        return "ArgusVideo";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeVideoFree(oldHandle);
    }

    /**
     * Prepares a video processing session from a file.
     */
    public static ArgusVideo loadFile(ArgusMultimodalContext mctx, Path filePath, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(mctx, "mctx");
        Objects.requireNonNull(filePath, "filePath");
        byte[] pathUtf8 = ArgusJniValidation.toUtf8NoNull(filePath.toAbsolutePath().toString(), "filePath");
        try (var mctxLease = mctx.lease()) {
            long ptr = ArgusJni.nativeVideoLoadFile(mctxLease.handle(), pathUtf8, fpsTarget, timestampIntervalMs);
            if (ptr == 0L) {
                ArgusNativeException.checkStatus(-1, "argus_video_load_file");
            }
            return new ArgusVideo(ptr, mctx);
        }
    }

    /**
     * Prepares a video processing session from an in-memory buffer.
     */
    public static ArgusVideo loadBuffer(ArgusMultimodalContext mctx, byte[] buffer, float fpsTarget, long timestampIntervalMs) {
        Objects.requireNonNull(mctx, "mctx");
        Objects.requireNonNull(buffer, "buffer");
        if (buffer.length == 0) {
            ArgusNativeException.checkStatus(-1, "argus_video_load_buffer");
        }
        try (var mctxLease = mctx.lease()) {
            long ptr = ArgusJni.nativeVideoLoadBuffer(mctxLease.handle(), buffer, 0, buffer.length, fpsTarget, timestampIntervalMs);
            if (ptr == 0L) {
                ArgusNativeException.checkStatus(-1, "argus_video_load_buffer");
            }
            return new ArgusVideo(ptr, mctx);
        }
    }

    public record VideoItem(ArgusBitmap bitmap, String text) {}

    @Deprecated
    public VideoItem readNext() {
        return withHandle(videoH -> {
            long[] outBitmapHandle = new long[1];
            byte[][] outTextBytes = new byte[1][];
            int res = ArgusJni.nativeVideoReadNext(videoH, outBitmapHandle, outTextBytes);
            if (res == 0) {
                long bmpH = outBitmapHandle[0];
                String text = ArgusJni.decodeUtf8(outTextBytes[0]);
                if (text != null && text.isEmpty()) {
                    text = null;
                }
                ArgusBitmap bitmap = bmpH != 0L ? new ArgusBitmap(bmpH) : null;
                return new VideoItem(bitmap, text);
            } else if (res == -1) {
                return null;
            } else {
                ArgusNativeException.checkStatus(res, "argus_video_read_next");
                return null;
            }
        });
    }

    /**
     * Reads the next frame bitmap or text timestamp from the video stream into the provided mutable carrier.
     */
    public boolean readNext(ArgusVideoItem item) {
        Objects.requireNonNull(item, "item");
        return withHandle(videoH -> {
            item.acquireWriteLock();
            try {
                long[] outBitmapHandle = item.outBitmapHandle();
                byte[][] outTextBytes = item.outTextBytes();
                outBitmapHandle[0] = 0L;
                outTextBytes[0] = null;

                int res = ArgusJni.nativeVideoReadNext(videoH, outBitmapHandle, outTextBytes);
                if (res == 0) {
                    long bmpH = outBitmapHandle[0];
                    String text = ArgusJni.decodeUtf8(outTextBytes[0]);
                    if (text != null && text.isEmpty()) {
                        text = null;
                    }
                    item.update(bmpH, text);
                    return true;
                } else if (res == -1) {
                    item.update(0L, null);
                    return false;
                } else {
                    ArgusNativeException.checkStatus(res, "argus_video_read_next");
                    return false;
                }
            } finally {
                item.releaseWriteLock();
            }
        });
    }
}
