package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Represents an active Whisper ASR transcribing session via the JDK 17+ / Android-compatible JNI bridge.
 * Extends {@link ArgusJniResource} for safe native leasing and idempotent lifecycle deallocation.
 */
public final class ArgusAudioContext extends ArgusJniResource {

    private ArgusAudioContext(long ctxHandle) {
        super(ctxHandle);
    }

    @Override
    protected String resourceName() {
        return "ArgusAudioContext";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeAudioFree(oldHandle);
    }

    /**
     * Initializes a Whisper transcribing session context.
     *
     * @param whisperModelPath path to the local Whisper GGUF weights
     * @param cpuThreads       CPU thread limit for tensor operations
     * @param gpuLayers        number of encoder/decoder layers to run in VRAM
     * @return an active ArgusAudioContext
     */
    public static ArgusAudioContext init(String whisperModelPath, int cpuThreads, int gpuLayers) {
        Objects.requireNonNull(whisperModelPath, "whisperModelPath");
        byte[] pathUtf8 = ArgusJniValidation.toUtf8NoNull(whisperModelPath, "whisperModelPath");
        long ptr = ArgusJni.nativeAudioInit(pathUtf8, cpuThreads, gpuLayers);
        if (ptr == 0L) {
            ArgusNativeException.throwLastError("argus_audio_init for " + whisperModelPath);
        }
        return new ArgusAudioContext(ptr);
    }

    /**
     * Synchronously transcribes a float PCM array containing 16kHz audio samples.
     */
    public String transcribe(float[] pcm, int sampleCount, int maxChars) {
        return transcribe(pcm, 0, sampleCount, maxChars);
    }

    /**
     * Synchronously transcribes a slice of a float PCM array containing 16kHz audio samples.
     */
    public String transcribe(float[] pcm, int offset, int sampleCount, int maxChars) {
        Objects.requireNonNull(pcm, "pcm");
        ArgusJniValidation.checkNonNegative(offset, "offset");
        ArgusJniValidation.checkNonNegative(sampleCount, "sampleCount");
        ArgusJniValidation.checkPositive(maxChars, "maxChars");
        if ((long) offset + (long) sampleCount > pcm.length) {
            throw new IllegalArgumentException(
                "PCM slice [offset=" + offset + ", sampleCount=" + sampleCount + ") exceeds pcm.length (" + pcm.length + ")"
            );
        }
        if (sampleCount == 0) {
            return "";
        }
        return withHandle(h -> {
            byte[] utf8 = ArgusJni.nativeTranscribeAudio(h, pcm, offset, sampleCount, maxChars);
            if (utf8 == null) {
                ArgusNativeException.throwLastError("argus_transcribe_audio");
            }
            return ArgusJni.decodeUtf8(utf8);
        });
    }

    /**
     * Synchronously transcribes a direct NIO ByteBuffer containing 16kHz float32 PCM samples.
     */
    public String transcribeDirect(ByteBuffer pcmDirectBuf, int sampleCount, int maxChars) {
        Objects.requireNonNull(pcmDirectBuf, "pcmDirectBuf");
        ArgusJniValidation.checkNonNegative(sampleCount, "sampleCount");
        ArgusJniValidation.checkPositive(maxChars, "maxChars");
        if (sampleCount == 0) {
            return "";
        }
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(sampleCount, Float.BYTES, "pcmDirectBuf");
        ArgusJniValidation.checkDirectBuffer(pcmDirectBuf, requiredBytes, "pcmDirectBuf");
        int byteOffset = pcmDirectBuf.position();
        if ((byteOffset & 3) != 0) {
            throw new IllegalArgumentException("pcmDirectBuf.position() (" + byteOffset + ") must be 4-byte aligned");
        }
        return withHandle(h -> {
            byte[] utf8 = ArgusJni.nativeTranscribeAudioDirectBuffer(h, pcmDirectBuf, byteOffset, sampleCount, maxChars);
            if (utf8 == null) {
                ArgusNativeException.throwLastError("argus_transcribe_audio");
            }
            return ArgusJni.decodeUtf8(utf8);
        });
    }

    public void setNThreads(int nThreads) {
        ArgusJniValidation.checkPositive(nThreads, "nThreads");
        withHandleVoid(h -> ArgusJni.nativeAudioSetNThreads(h, nThreads));
    }

    public int getNThreads() {
        return withHandle(ArgusJni::nativeAudioGetNThreads);
    }
}
