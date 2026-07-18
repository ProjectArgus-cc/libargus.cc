package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * High-level object representing an active multimodal projector context weights/session.
 * Implements AutoCloseable for safe resource deallocation in unmanaged space.
 */
public final class ArgusMultimodalContext implements AutoCloseable {
    private MemorySegment mctxPtr;
    private final ArgusModel modelRef;

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
                throw new RuntimeException("Native argus_multimodal_init returned NULL for: " + mmprojPath);
            }
            return new ArgusMultimodalContext(mctxPtr, model);
        } catch (Throwable t) {
            model.release();
            throw new RuntimeException("Failed to load native multimodal projector context", t);
        }
    }

    public boolean supportVision() {
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_vision.invokeExact(mctxPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to check vision support", t);
        }
    }

    public boolean supportAudio() {
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_audio.invokeExact(mctxPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to check audio support", t);
        }
    }

    public boolean supportVideo() {
        try {
            return (boolean) ArgusBindings.argus_multimodal_support_video.invokeExact(mctxPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to check video support", t);
        }
    }

    public int getAudioSampleRate() {
        try {
            return (int) ArgusBindings.argus_multimodal_get_audio_sample_rate.invokeExact(mctxPtr);
        } catch (Throwable t) {
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

        ArgusInputChunks chunks = ArgusInputChunks.init();
        try {
            MemorySegment textSeg = arena.allocateFrom(text);
            MemorySegment bitmapsArray = MemorySegment.NULL;
            if (!bitmaps.isEmpty()) {
                bitmapsArray = arena.allocate(ValueLayout.ADDRESS, bitmaps.size());
                for (int i = 0; i < bitmaps.size(); ++i) {
                    bitmapsArray.setAtIndex(ValueLayout.ADDRESS, i, bitmaps.get(i).getHandle());
                }
            }

            int res = (int) ArgusBindings.argus_multimodal_tokenize.invokeExact(
                mctxPtr,
                chunks.getHandle(),
                textSeg,
                addBos,
                bitmapsArray,
                bitmaps.size()
            );

            if (res != 0) {
                chunks.close();
                throw new RuntimeException("Native argus_multimodal_tokenize returned error code: " + res);
            }

            return chunks;
        } catch (Throwable t) {
            chunks.close();
            throw new RuntimeException("Failed to execute native multimodal tokenization", t);
        }
    }

    public MemorySegment getHandle() {
        return mctxPtr;
    }

    @Override
    public synchronized void close() {
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
