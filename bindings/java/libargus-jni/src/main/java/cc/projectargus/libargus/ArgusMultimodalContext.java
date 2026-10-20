package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * High-level object representing an active multimodal projector context weights/session
 * via the JDK 17+ / Android-compatible JNI bridge.
 */
public final class ArgusMultimodalContext extends ArgusJniResource {
    private final ArgusModel modelRef;

    private ArgusMultimodalContext(long mctxHandle, ArgusModel modelRef) {
        super(mctxHandle);
        this.modelRef = Objects.requireNonNull(modelRef, "modelRef");
    }

    @Override
    protected String resourceName() {
        return "ArgusMultimodalContext";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeMultimodalFree(oldHandle);
    }

    /**
     * Initializes and loads the multimodal projector model.
     */
    public static ArgusMultimodalContext init(ArgusModel model, Path mmprojPath, int cpuThreads, boolean useGpu) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        byte[] pathUtf8 = ArgusJniValidation.toUtf8NoNull(mmprojPath.toAbsolutePath().toString(), "mmprojPath");
        try (var modelLease = model.lease()) {
            long ptr = ArgusJni.nativeMultimodalInit(modelLease.handle(), pathUtf8, cpuThreads, useGpu);
            if (ptr == 0L) {
                ArgusNativeException.throwLastError("argus_multimodal_init for " + mmprojPath);
            }
            return new ArgusMultimodalContext(ptr, model);
        }
    }

    public boolean supportVision() {
        return withHandle(ArgusJni::nativeMultimodalSupportVision);
    }

    public boolean supportAudio() {
        return withHandle(ArgusJni::nativeMultimodalSupportAudio);
    }

    public boolean supportVideo() {
        return withHandle(ArgusJni::nativeMultimodalSupportVideo);
    }

    public int getAudioSampleRate() {
        return withHandle(ArgusJni::nativeMultimodalGetAudioSampleRate);
    }

    /**
     * Tokenizes prompt text and media bitmaps into a sequential chunk container.
     */
    public ArgusInputChunks tokenize(String text, boolean addBos, List<ArgusBitmap> bitmaps) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(bitmaps, "bitmaps");
        byte[] textUtf8 = ArgusJniValidation.toUtf8(text, "text");
        return tokenize(textUtf8, addBos, bitmaps);
    }

    /**
     * Tokenizes UTF-8 prompt bytes and media bitmaps into a sequential chunk container.
     */
    public ArgusInputChunks tokenize(byte[] textUtf8, boolean addBos, List<ArgusBitmap> bitmaps) {
        Objects.requireNonNull(textUtf8, "textUtf8");
        Objects.requireNonNull(bitmaps, "bitmaps");

        return withHandle(mctxH -> {
            ArgusInputChunks chunks = ArgusInputChunks.init();
            try (var chunksLease = chunks.lease()) {
                int nBitmaps = bitmaps.size();
                long[] bitmapHandles = nBitmaps > 0 ? new long[nBitmaps] : null;
                ArgusJniResource.Lease[] leasedBitmaps = new ArgusJniResource.Lease[nBitmaps];
                try {
                    for (int i = 0; i < nBitmaps; i++) {
                        leasedBitmaps[i] = Objects.requireNonNull(bitmaps.get(i), "bitmap[" + i + "]").lease();
                        bitmapHandles[i] = leasedBitmaps[i].handle();
                    }
                    int res = ArgusJni.nativeMultimodalTokenize(
                        mctxH,
                        chunksLease.handle(),
                        textUtf8,
                        addBos,
                        bitmapHandles,
                        nBitmaps
                    );
                    if (res != 0) {
                        ArgusNativeException.checkStatus(res, "argus_multimodal_tokenize_n");
                    }
                    return chunks;
                } finally {
                    for (int i = 0; i < nBitmaps; i++) {
                        if (leasedBitmaps[i] != null) {
                            try {
                                leasedBitmaps[i].close();
                            } catch (Throwable ignored) {}
                        }
                    }
                }
            } catch (Throwable t) {
                chunks.close();
                if (t instanceof RuntimeException re) throw re;
                throw new RuntimeException("Failed to execute native multimodal tokenization", t);
            }
        });
    }
}
