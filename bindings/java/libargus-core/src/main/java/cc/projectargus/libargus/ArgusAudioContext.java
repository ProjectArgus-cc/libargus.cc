package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Represents an active Whisper ASR transcribing session.
 * Implements AutoCloseable to ensure native resources are safely released.
 */
public final class ArgusAudioContext implements AutoCloseable {
    private MemorySegment ctxPtr;

    private ArgusAudioContext(MemorySegment ctxPtr) {
        this.ctxPtr = Objects.requireNonNull(ctxPtr);
    }

    /**
     * Initializes a Whisper transcribing session context.
     *
     * @param arena            allocation scope for configurations
     * @param whisperModelPath path to the local Whisper GGUF weights
     * @param cpuThreads       CPU thread limit for tensor operations
     * @param gpuLayers        number of encoder/decoder layers to run in VRAM
     * @return an active ArgusAudioContext
     */
    public static ArgusAudioContext init(Arena arena, String whisperModelPath, int cpuThreads, int gpuLayers) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(whisperModelPath);

        MemorySegment pathSeg = arena.allocateFrom(whisperModelPath);
        MemorySegment paramsSeg = arena.allocate(ArgusLayouts.AUDIO_PARAMS);

        paramsSeg.set(ValueLayout.ADDRESS, 
            ArgusLayouts.AUDIO_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("whisper_model_path")), 
            pathSeg
        );
        paramsSeg.set(ValueLayout.JAVA_INT, 
            ArgusLayouts.AUDIO_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("cpu_threads")), 
            cpuThreads
        );
        paramsSeg.set(ValueLayout.JAVA_INT, 
            ArgusLayouts.AUDIO_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("gpu_layers")), 
            gpuLayers
        );

        try {
            MemorySegment ctxPtr = (MemorySegment) ArgusBindings.argus_audio_init.invokeExact(paramsSeg);
            if (ctxPtr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_audio_init returned NULL for path: " + whisperModelPath);
            }
            return new ArgusAudioContext(ctxPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute native audio initialization", t);
        }
    }

    /**
     * Synchronously transcribes a float PCM buffer containing 16kHz audio samples.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     *
     * @param pcmSeg      audio float data normalized to [-1.0, 1.0] (16kHz sampling rate)
     * @param sampleCount number of float samples in the PCM buffer
     * @param maxChars    maximum output character capacity to allocate off-heap
     * @return transcribed text string
     */
    public String transcribe(MemorySegment pcmSeg, int sampleCount, int maxChars) {
        Objects.requireNonNull(pcmSeg);
        if (sampleCount == 0 || maxChars <= 0) {
            return "";
        }

        try (Arena local = Arena.ofConfined()) {
            MemorySegment textSeg = local.allocate(ValueLayout.JAVA_BYTE, maxChars);

            int result = (int) ArgusBindings.argus_transcribe_audio.invokeExact(
                ctxPtr,
                pcmSeg,
                sampleCount,
                textSeg,
                maxChars
            );

            if (result < 0) {
                throw new RuntimeException("Native transcription failed with error code: " + result);
            }

            return textSeg.getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to run audio transcribing", t);
        }
    }

    /**
     * Sets the CPU thread count for this Whisper audio context session.
     *
     * @param nThreads number of threads for acoustic matrix calculations
     */
    public void setNThreads(int nThreads) {
        if (ctxPtr == null || ctxPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Audio context session has been closed");
        }
        try {
            ArgusBindings.argus_audio_set_n_threads.invokeExact(ctxPtr, nThreads);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to set thread count for audio context", t);
        }
    }

    /**
     * Queries active CPU thread count for this Whisper audio context session.
     *
     * @return number of allocated acoustic calculation threads
     */
    public int getNThreads() {
        if (ctxPtr == null || ctxPtr.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Audio context session has been closed");
        }
        try {
            return (int) ArgusBindings.argus_audio_get_n_threads.invokeExact(ctxPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to query thread count for audio context", t);
        }
    }

    /**
     * Returns the raw memory address representing the unmanaged context structure.
     */
    public MemorySegment getHandle() {
        return ctxPtr;
    }


    @Override
    public synchronized void close() {
        if (ctxPtr != null && !ctxPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_audio_free.invokeExact(ctxPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native audio context resources", t);
            } finally {
                ctxPtr = MemorySegment.NULL;
            }
        }
    }
}
