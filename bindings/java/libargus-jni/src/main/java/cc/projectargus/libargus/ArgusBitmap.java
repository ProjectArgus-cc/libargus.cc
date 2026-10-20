package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wraps raw/parsed media content in native memory via the JDK 17+ / Android-compatible JNI bridge.
 */
public final class ArgusBitmap extends ArgusJniResource {

    ArgusBitmap(long bitmapHandle) {
        super(bitmapHandle);
    }

    @Override
    protected String resourceName() {
        return "ArgusBitmap";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeBitmapFree(oldHandle);
    }

    /**
     * Creates a bitmap from a raw RGB byte array (RGBRGB... format).
     */
    public static ArgusBitmap fromRgb(int width, int height, byte[] rgbData) {
        Objects.requireNonNull(rgbData, "rgbData");
        ArgusJniValidation.checkPositive(width, "width");
        ArgusJniValidation.checkPositive(height, "height");
        long numPixels = ArgusJniValidation.multiplyExactBytes(width, height, "pixels");
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(numPixels, 3, "rgbData");
        if (rgbData.length < requiredBytes) {
            throw new IllegalArgumentException("rgbData.length (" + rgbData.length + ") < requiredBytes (" + requiredBytes + ")");
        }
        long ptr = ArgusJni.nativeBitmapFromRgb(width, height, rgbData);
        if (ptr == 0L) {
            ArgusNativeException.checkStatus(-1, "argus_bitmap_from_rgb");
        }
        return new ArgusBitmap(ptr);
    }

    /**
     * Creates a bitmap from a direct NIO ByteBuffer containing raw RGB pixels (RGBRGB... format).
     */
    public static ArgusBitmap fromRgbDirect(int width, int height, ByteBuffer rgbDirectBuf) {
        Objects.requireNonNull(rgbDirectBuf, "rgbDirectBuf");
        ArgusJniValidation.checkPositive(width, "width");
        ArgusJniValidation.checkPositive(height, "height");
        long numPixels = ArgusJniValidation.multiplyExactBytes(width, height, "pixels");
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(numPixels, 3, "rgbDirectBuf");
        ArgusJniValidation.checkDirectBuffer(rgbDirectBuf, requiredBytes, "rgbDirectBuf");
        long ptr = ArgusJni.nativeBitmapFromRgbDirectBuffer(width, height, rgbDirectBuf, rgbDirectBuf.position());
        if (ptr == 0L) {
            ArgusNativeException.checkStatus(-1, "argus_bitmap_from_rgb");
        }
        return new ArgusBitmap(ptr);
    }

    /**
     * Creates a bitmap from a raw float PCM audio array (e.g. 16kHz).
     */
    public static ArgusBitmap fromPcm(float[] pcmData, int nSamples) {
        Objects.requireNonNull(pcmData, "pcmData");
        ArgusJniValidation.checkPositive(nSamples, "nSamples");
        if (nSamples > pcmData.length) {
            throw new IllegalArgumentException("nSamples (" + nSamples + ") > pcmData.length (" + pcmData.length + ")");
        }
        long ptr = ArgusJni.nativeBitmapFromPcm(pcmData, nSamples);
        if (ptr == 0L) {
            ArgusNativeException.checkStatus(-1, "argus_bitmap_from_pcm");
        }
        return new ArgusBitmap(ptr);
    }

    /**
     * Creates a bitmap from a direct NIO ByteBuffer containing float32 PCM samples.
     */
    public static ArgusBitmap fromPcmDirect(ByteBuffer pcmDirectBuf, int nSamples) {
        Objects.requireNonNull(pcmDirectBuf, "pcmDirectBuf");
        ArgusJniValidation.checkPositive(nSamples, "nSamples");
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(nSamples, Float.BYTES, "pcmDirectBuf");
        ArgusJniValidation.checkDirectBuffer(pcmDirectBuf, requiredBytes, "pcmDirectBuf");
        int byteOffset = pcmDirectBuf.position();
        if ((byteOffset & 3) != 0) {
            throw new IllegalArgumentException("pcmDirectBuf.position() (" + byteOffset + ") must be 4-byte aligned");
        }
        long ptr = ArgusJni.nativeBitmapFromPcmDirectBuffer(pcmDirectBuf, byteOffset, nSamples);
        if (ptr == 0L) {
            ArgusNativeException.checkStatus(-1, "argus_bitmap_from_pcm");
        }
        return new ArgusBitmap(ptr);
    }

    /**
     * Automatically loads and processes media from a local file path.
     */
    public static ArgusBitmap loadFile(ArgusMultimodalContext mctx, Path filePath, boolean placeholder) {
        Objects.requireNonNull(mctx, "mctx");
        Objects.requireNonNull(filePath, "filePath");
        byte[] pathUtf8 = ArgusJniValidation.toUtf8NoNull(filePath.toAbsolutePath().toString(), "filePath");
        try (var mctxLease = mctx.lease()) {
            long ptr = ArgusJni.nativeBitmapLoadFile(mctxLease.handle(), pathUtf8, placeholder);
            if (ptr == 0L) {
                return null;
            }
            return new ArgusBitmap(ptr);
        }
    }

    /**
     * Automatically loads and processes media from a memory buffer.
     */
    public static ArgusBitmap loadBuffer(ArgusMultimodalContext mctx, byte[] buffer, boolean placeholder) {
        Objects.requireNonNull(mctx, "mctx");
        Objects.requireNonNull(buffer, "buffer");
        if (buffer.length == 0) {
            return null;
        }
        try (var mctxLease = mctx.lease()) {
            long ptr = ArgusJni.nativeBitmapLoadBuffer(mctxLease.handle(), buffer, 0, buffer.length, placeholder);
            if (ptr == 0L) {
                return null;
            }
            return new ArgusBitmap(ptr);
        }
    }

    /**
     * Configures whether this bitmap is temporally mergeable by video models during tokenization.
     */
    public void setMergeable(boolean mergeable) {
        withHandleVoid(h -> {
            if (!ArgusJni.nativeBitmapSetMergeable(h, mergeable)) {
                ArgusNativeException.throwLastError("argus_bitmap_set_mergeable");
            }
        });
    }
}
