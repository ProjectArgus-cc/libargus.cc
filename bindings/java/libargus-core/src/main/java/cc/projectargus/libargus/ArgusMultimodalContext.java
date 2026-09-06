package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level object representing an active multimodal projector context weights/session.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusMultimodalContext implements AutoCloseable {
    private MemorySegment mctxPtr;
    private final ArgusModel modelRef;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private ArgusMultimodalContext(MemorySegment mctxPtr, ArgusModel modelRef) {
        this.mctxPtr = Objects.requireNonNull(mctxPtr);
        this.modelRef = Objects.requireNonNull(modelRef);
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

        model.acquire();

        try {
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

            MemorySegment mctxPtr = (MemorySegment) ArgusBindings.argus_multimodal_init.invokeExact(model.getHandle(), paramsSeg);
            if (mctxPtr.equals(MemorySegment.NULL)) {
                ArgusNativeException.checkStatus(-1, "argus_multimodal_init");
            }
            return new ArgusMultimodalContext(mctxPtr, model);
        } catch (Throwable t) {
            model.release();
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to load native multimodal projector context", t);
        }
    }

    public boolean supportVision() {
        checkNotClosed();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_vision.invokeExact(mctxPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check vision support", t);
        }
    }

    public boolean supportAudio() {
        checkNotClosed();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_audio.invokeExact(mctxPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check audio support", t);
        }
    }

    public boolean supportVideo() {
        checkNotClosed();
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_video.invokeExact(mctxPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check video support", t);
        }
    }

    public int getAudioSampleRate() {
        checkNotClosed();
        try {
            return (int) ArgusBindings.argus_multimodal_get_audio_sample_rate.invokeExact(mctxPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to get audio sample rate", t);
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

        checkNotClosed();

        ArgusInputChunks chunks = ArgusInputChunks.init();
        try {
            MemorySegment bitmapsArray = MemorySegment.NULL;
            int nBitmaps = bitmaps.size();
            if (nBitmaps > 0) {
                bitmapsArray = arena.allocate(ValueLayout.ADDRESS, nBitmaps);
                for (int i = 0; i < nBitmaps; ++i) {
                    bitmapsArray.setAtIndex(ValueLayout.ADDRESS, i, bitmaps.get(i).getHandle());
                }
            }

            int res = (int) ArgusBindings.argus_multimodal_tokenize_n.invokeExact(
                mctxPtr,
                chunks.getHandle(),
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
        } catch (Throwable t) {
            chunks.close();
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to execute native multimodal tokenization", t);
        }
    }

    public MemorySegment getHandle() {
        checkNotClosed();
        return mctxPtr;
    }

    public boolean isClosed() {
        return closed.get();
    }

    private void checkNotClosed() {
        if (closed.get() || mctxPtr == null || mctxPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Multimodal context session has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (mctxPtr != null && !mctxPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_multimodal_free.invokeExact(mctxPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native multimodal projector context", t);
            } finally {
                mctxPtr = MemorySegment.NULL;
                modelRef.release();
            }
        }
    }
}
