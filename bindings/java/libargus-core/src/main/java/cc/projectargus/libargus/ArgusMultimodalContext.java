package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * High-level object representing an active multimodal projector context weights/session.
 * Extends {@link ArgusNativeResource} for safe native leasing and idempotent lifecycle management.
 */
public final class ArgusMultimodalContext extends ArgusNativeResource {
    private final ArgusModel modelRef;

    private ArgusMultimodalContext(MemorySegment mctxPtr, ArgusModel modelRef) {
        super(mctxPtr);
        this.modelRef = Objects.requireNonNull(modelRef);
    }

    @Override
    protected String resourceName() {
        return "ArgusMultimodalContext";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_multimodal_free.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        }
    }

    /**
     * Initializes and loads the multimodal projector model.
     *
     * @param arena      the allocation scope for setup parameters
     * @param model      the loaded base language model
     * @param mmprojPath path to the projector GGUF file
     * @param cpuThreads compute thread count for projection pass
     * @param useGpu     enable GPU offloading for projector weights
     * @return a loaded ArgusMultimodalContext
     */
    public static ArgusMultimodalContext init(Arena arena, ArgusModel model, Path mmprojPath, int cpuThreads, boolean useGpu) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(model);
        Objects.requireNonNull(mmprojPath);

        try (var modelLease = model.lease()) {
            MemorySegment pathSeg = arena.allocateFrom(mmprojPath.toAbsolutePath().toString());
            MemorySegment paramsSeg = arena.allocate(ArgusLayouts.MULTIMODAL_PARAMS);

            paramsSeg.set(ValueLayout.ADDRESS, 
                ArgusLayouts.MULTIMODAL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("mmproj_path")), 
                pathSeg
            );
            paramsSeg.set(ValueLayout.JAVA_INT, 
                ArgusLayouts.MULTIMODAL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("cpu_threads")), 
                cpuThreads
            );
            paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                ArgusLayouts.MULTIMODAL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("use_gpu")), 
                useGpu
            );

            MemorySegment mctxPtr = (MemorySegment) ArgusBindings.argus_multimodal_init.invokeExact(modelLease.handle(), paramsSeg);
            if (mctxPtr.equals(MemorySegment.NULL)) {
                ArgusNativeException.throwLastError("argus_multimodal_init for " + mmprojPath);
            }
            return new ArgusMultimodalContext(mctxPtr, model);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load native multimodal projector context", t);
        }
    }

    public boolean supportVision() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_vision.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check vision support", t);
        } finally {
            releaseReadLease();
        }
    }

    public boolean supportAudio() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_audio.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check audio support", t);
        } finally {
            releaseReadLease();
        }
    }

    public boolean supportVideo() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_video.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check video support", t);
        } finally {
            releaseReadLease();
        }
    }

    public int getAudioSampleRate() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_multimodal_get_audio_sample_rate.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to get audio sample rate", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Tokenizes prompt text and media bitmaps into a sequential chunk container.
     */
    public ArgusInputChunks tokenize(Arena arena, String text, boolean addBos, List<ArgusBitmap> bitmaps) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(text);
        Objects.requireNonNull(bitmaps);
        byte[] textBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        MemorySegment textSeg = arena.allocateFrom(ValueLayout.JAVA_BYTE, textBytes);
        return tokenize(arena, textSeg, textBytes.length, addBos, bitmaps);
    }

    /**
     * Tokenizes prompt text segment and media bitmaps into a sequential chunk container using exact byte length.
     */
    public ArgusInputChunks tokenize(Arena arena, MemorySegment textSeg, long textLen, boolean addBos, List<ArgusBitmap> bitmaps) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(bitmaps);
        ArgusValidation.checkNonNegative(textLen, "textLen");
        ArgusValidation.checkReadable(textSeg, textLen, "textSeg");

        MemorySegment mctxH = acquireReadLease();
        try {
            ArgusInputChunks chunks = ArgusInputChunks.init();
            try (var lChunks = chunks.lease()) {
                int nBitmaps = bitmaps.size();
                MemorySegment bitmapsArray = MemorySegment.NULL;
                ArgusNativeResource.Lease[] leasedBitmaps = new ArgusNativeResource.Lease[nBitmaps];
                try {
                    for (int i = 0; i < nBitmaps; i++) {
                        leasedBitmaps[i] = bitmaps.get(i).lease();
                    }
                    if (nBitmaps > 0) {
                        bitmapsArray = arena.allocate(ValueLayout.ADDRESS, nBitmaps);
                        for (int i = 0; i < nBitmaps; ++i) {
                            bitmapsArray.setAtIndex(ValueLayout.ADDRESS, i, leasedBitmaps[i].handle());
                        }
                    }

                    int res = (int) ArgusBindings.argus_multimodal_tokenize_n.invokeExact(
                        mctxH,
                        lChunks.handle(),
                        textSeg,
                        textLen,
                        addBos,
                        bitmapsArray,
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
                            } catch (Throwable ignored) {
                            }
                        }
                    }
                }
            }
 catch (Throwable t) {
                chunks.close();
                if (t instanceof RuntimeException re) throw re;
                throw new RuntimeException("Failed to execute native multimodal tokenization", t);
            }
        } finally {
            releaseReadLease();
        }
    }
}
