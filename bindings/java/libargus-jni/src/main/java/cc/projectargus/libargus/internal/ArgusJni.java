package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusLogCallback;
import cc.projectargus.libargus.ArgusLogLevel;
import cc.projectargus.libargus.ArgusNativeException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * Low-level JNI bridge to the unified {@code libargus} native shared library.
 * Delegates native library discovery and SPI extraction to {@link ArgusNativeLoader}.
 *
 * <p>All string parameters and return values are transported as Standard UTF-8 {@code byte[]}
 * arrays rather than JNI Modified UTF-8 {@code jstring} to guarantee lossless 4-byte code points,
 * prevent Android ART CheckJNI surrogate rejections, and preserve raw BPE token byte slices.
 */
public final class ArgusJni {
    public static final String VERSION;
    public static final String EXTRACTED_DIR;

    private static volatile ArgusLogCallback activeLogCallback = null;
    private static final ThreadLocal<Boolean> IN_LOG_CALLBACK = ThreadLocal.withInitial(() -> Boolean.FALSE);

    static {
        ArgusNativeLoader.ensureLoaded();
        VERSION = ArgusNativeLoader.VERSION;
        EXTRACTED_DIR = ArgusNativeLoader.EXTRACTED_DIR;
        registerErrorResolver();
    }

    private ArgusJni() {}

    public static String loadedLibraryPath() {
        return ArgusNativeLoader.loadedLibraryPath();
    }

    public static String loadedProvider() {
        return ArgusNativeLoader.loadedProvider();
    }

    public static String decodeUtf8(byte[] utf8Bytes) {
        if (utf8Bytes == null) {
            return null;
        }
        if (utf8Bytes.length == 0) {
            return "";
        }
        return new String(utf8Bytes, StandardCharsets.UTF_8);
    }

    /**
     * Upcall entry point invoked by {@code jni_log_trampoline} in {@code argus_jni.cc}.
     */
    @SuppressWarnings("unused")
    public static void onNativeLog(int level, byte[] utf8Bytes) {
        ArgusLogCallback cb = activeLogCallback;
        if (cb == null || utf8Bytes == null) {
            return;
        }
        IN_LOG_CALLBACK.set(Boolean.TRUE);
        try {
            String msg = new String(utf8Bytes, StandardCharsets.UTF_8);
            cb.onLog(ArgusLogLevel.fromValue(level), msg);
        } catch (Throwable ignored) {
            // Protect native callers from Java callback exceptions
        } finally {
            IN_LOG_CALLBACK.set(Boolean.FALSE);
        }
    }

    public static boolean isInLogCallback() {
        return Boolean.TRUE.equals(IN_LOG_CALLBACK.get());
    }

    public static synchronized void setLogCallback(ArgusLogCallback callback) {
        activeLogCallback = callback;
        nativeSetLogCallbackEnabled(callback != null);
    }

    // =========================================================================
    // 1. Diagnostics, Build Identity, Logging & Backend Lifecycle
    // =========================================================================

    public static native int nativeLastErrorCode();
    public static native byte[] nativeLastErrorMessage();
    public static native void nativeClearError();
    public static native long nativeBuildFeatures();
    public static native byte[] nativeBuildInfo();
    public static native boolean nativeBackendInit(byte[] customPluginPathUtf8);
    public static native void nativeBackendFree();
    public static native boolean nativeBackendIsInitialized();
    public static native int nativeBackendGetCount();
    public static native byte[] nativeBackendGetName(int index);
    public static native byte[] nativeVersion();
    public static native void nativeSetLogLevel(int level);
    public static native int nativeGetLogLevel();
    public static native void nativeSetLogCallbackEnabled(boolean enabled);
    public static native int nativeLogInFlightCount();

    // =========================================================================
    // 2. Model Lifecycle, Vocabulary & GGUF Introspection
    // =========================================================================

    public static native long nativeModelLoad(byte[] pathUtf8, int gpuLayers, boolean useMlock);
    public static native long nativeModelLoadFd(int fd, int gpuLayers, boolean useMlock);
    public static native boolean nativeModelRetain(long modelHandle);
    public static native void nativeModelRelease(long modelHandle);
    public static native void nativeModelFree(long modelHandle);

    public static native int nativeTokenize(
        long modelHandle,
        byte[] utf8Text,
        int[] outTokens,
        int maxTokens,
        boolean addBos
    );
    public static native byte[] nativeTokenToPiece(long modelHandle, int token);
    public static native int nativeTokenToPieceInto(
        long modelHandle,
        int token,
        byte[] outBuf,
        int offset,
        int length
    );

    public static native int nativeVocabBos(long modelHandle);
    public static native int nativeVocabEos(long modelHandle);
    public static native int nativeVocabEot(long modelHandle);
    public static native int nativeVocabPad(long modelHandle);
    public static native int nativeVocabNTokens(long modelHandle);
    public static native boolean nativeVocabIsEog(long modelHandle, int token);

    public static native byte[] nativeModelMetaValStr(long modelHandle, byte[] keyUtf8);
    public static native int nativeModelMetaCount(long modelHandle);
    public static native byte[] nativeModelMetaKeyByIndex(long modelHandle, int index);
    public static native byte[] nativeModelMetaValStrByIndex(long modelHandle, int index);

    public static native int nativeModelNEmbd(long modelHandle);
    public static native int nativeModelNEmbdOut(long modelHandle);
    public static native int nativeModelNEmbdInp(long modelHandle);
    public static native int nativeModelNCtxTrain(long modelHandle);
    public static native int nativeModelNLayer(long modelHandle);
    public static native int nativeModelNHead(long modelHandle);
    public static native int nativeModelNHeadKv(long modelHandle);
    public static native long nativeModelNParams(long modelHandle);
    public static native boolean nativeModelHasEncoder(long modelHandle);
    public static native int nativeModelNPosPerEmbd(long modelHandle);
    public static native boolean nativeModelIsMRoPE(long modelHandle);
    public static native boolean nativeModelIsRecurrent(long modelHandle);
    public static native boolean nativeModelIsHybrid(long modelHandle);
    public static native boolean nativeModelIsDiffusion(long modelHandle);
    public static native long nativeModelSize(long modelHandle);
    public static native byte[] nativeModelDesc(long modelHandle);
    public static native long nativeModelKvBytesPerToken(long modelHandle, int typeK, int typeV);
    public static native long nativeModelEstimateVramBytes(long modelHandle, int contextLength, int typeK, int typeV);
    public static native long nativeQuantTypeSize(int type);
    public static native int nativeQuantBlockSize(int type);

    // =========================================================================
    // 3. Context Lifecycle, Telemetry, Decoding, Embeddings, Sampling & KV
    // =========================================================================

    public static native long nativeContextInit(
        long modelHandle,
        long draftModelHandle,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        int uBatch,
        int nSeqMax,
        boolean enableDraftMtp,
        boolean embeddings,
        boolean kvUnified
    );
    public static native void nativeContextFree(long ctxHandle);
    public static native long nativeContextGetModel(long ctxHandle);
    public static native void nativeSetNThreads(long ctxHandle, int nThreads, int nThreadsBatch);
    public static native int nativeGetNThreads(long ctxHandle);
    public static native int nativeGetNThreadsBatch(long ctxHandle);
    public static native boolean nativeContextGetPerf(long ctxHandle, double[] outDoubles, int[] outInts);
    public static native void nativeContextResetPerf(long ctxHandle);
    public static native boolean nativeContextSetCausalAttn(long ctxHandle, boolean causal);
    public static native int nativeContextGetCausalAttn(long ctxHandle);
    public static native boolean nativeContextHasDraft(long ctxHandle);
    public static native int nativeContextGetSeqMax(long ctxHandle);

    public static native int nativeDecodeTokensArray(
        long ctxHandle,
        int[] tokens,
        int offset,
        int nTokens,
        int startPos,
        int seqId,
        boolean requestLogits,
        long abortFlagHandle
    );
    public static native int nativeDecodeTokensDirectBuffer(
        long ctxHandle,
        ByteBuffer directBuf,
        int byteOffset,
        int nTokens,
        int startPos,
        int seqId,
        boolean requestLogits,
        long abortFlagHandle
    );

    public static native int nativeGetEmbeddings(long ctxHandle, int seqId, float[] outEmbeddings, int maxFloats);
    public static native int nativeGetEmbeddingsDirectBuffer(
        long ctxHandle,
        int seqId,
        ByteBuffer directBuf,
        int byteOffset,
        int maxFloats
    );

    public static native int nativeEvalEmbeddings(
        long ctxHandle,
        int[] tokens,
        int offset,
        int nTokens,
        int seqId,
        float[] outEmbeddings,
        int maxFloats
    );
    public static native int nativeEvalEmbeddingsDirectBuffer(
        long ctxHandle,
        ByteBuffer tokensBuf,
        int tokensByteOffset,
        int nTokens,
        int seqId,
        ByteBuffer outEmbeddingsBuf,
        int outByteOffset,
        int maxFloats
    );

    public static native int nativeSampleToken(long ctxHandle, int seqId, float temperature, float repeatPenalty);
    public static native int nativeSampleTokenWithBias(
        long ctxHandle,
        int seqId,
        float temperature,
        float repeatPenalty,
        int[] biasTokens,
        float[] biasValues,
        int biasCount
    );
    public static native int nativeSampleTokenExt(
        long ctxHandle,
        int seqId,
        float temperature,
        float repeatPenalty,
        int repeatLastN,
        float frequencyPenalty,
        float presencePenalty,
        float topP,
        float minP,
        int topK,
        float dryMultiplier,
        float dryBase,
        int dryAllowedLength,
        int dryPenaltyLastN,
        int seed,
        int[] biasTokens,
        float[] biasValues,
        int biasCount
    );

    public static native int nativeSamplerReset(long ctxHandle, int seqId);
    public static native int nativeSamplerPrime(long ctxHandle, int seqId, int[] tokens, int offset, int nTokens);
    public static native int nativeSamplerTruncate(long ctxHandle, int seqId, int newLength);
    public static native int nativeSamplerGetHistoryCount(long ctxHandle, int seqId);
    public static native int nativeSamplerHasPending(long ctxHandle, int seqId);
    public static native int nativeSamplerDiscardPending(long ctxHandle, int seqId);

    public static native boolean nativeKvCacheClearSlot(long ctxHandle, int seqId, int p0, int p1);
    public static native boolean nativeKvCacheCanShift(long ctxHandle);
    public static native boolean nativeKvCacheSeqAdd(long ctxHandle, int seqId, int p0, int p1, int delta);
    public static native boolean nativeKvCacheSeqCp(long ctxHandle, int seqIdSrc, int seqIdDst, int p0, int p1);
    public static native int nativeKvCacheSeqPosMax(long ctxHandle, int seqId);
    public static native int nativeKvCacheSeqPosMin(long ctxHandle, int seqId);

    public static native int nativeSynthesizeSpeech(
        long ctxHandle,
        long vocoderModelHandle,
        byte[] utf8Text,
        int voiceSeed,
        float[] outPcm,
        int maxSamples
    );

    // =========================================================================
    // 4. Atomic Cancellation Controls (ArgusAbortFlag)
    // =========================================================================

    public static native long nativeAbortFlagCreate();
    public static native void nativeAbortFlagRequest(long flagHandle);
    public static native void nativeAbortFlagReset(long flagHandle);
    public static native boolean nativeAbortFlagIsRequested(long flagHandle);
    public static native boolean nativeAbortFlagRetain(long flagHandle);
    public static native void nativeAbortFlagRelease(long flagHandle);

    // =========================================================================
    // 5. Audio Stream Transcription Subsystem (Whisper ASR)
    // =========================================================================

    public static native long nativeAudioInit(byte[] whisperModelPathUtf8, int cpuThreads, int gpuLayers);
    public static native void nativeAudioFree(long audioCtxHandle);
    public static native void nativeAudioSetNThreads(long audioCtxHandle, int nThreads);
    public static native int nativeAudioGetNThreads(long audioCtxHandle);
    public static native byte[] nativeTranscribeAudio(
        long audioCtxHandle,
        float[] pcmData,
        int offset,
        int sampleCount,
        int maxChars
    );
    public static native byte[] nativeTranscribeAudioDirectBuffer(
        long audioCtxHandle,
        ByteBuffer pcmBuf,
        int byteOffset,
        int sampleCount,
        int maxChars
    );

    // =========================================================================
    // 6. Multimodal Vision / Audio / Video Operations
    // =========================================================================

    public static native long nativeMultimodalInit(long modelHandle, byte[] mmprojPathUtf8, int cpuThreads, boolean useGpu);
    public static native boolean nativeMultimodalRetain(long mctxHandle);
    public static native void nativeMultimodalRelease(long mctxHandle);
    public static native void nativeMultimodalFree(long mctxHandle);
    public static native boolean nativeMultimodalSupportVision(long mctxHandle);
    public static native boolean nativeMultimodalSupportAudio(long mctxHandle);
    public static native boolean nativeMultimodalSupportVideo(long mctxHandle);
    public static native int nativeMultimodalGetAudioSampleRate(long mctxHandle);

    public static native long nativeBitmapFromRgb(int width, int height, byte[] rgbData);
    public static native long nativeBitmapFromRgbDirectBuffer(int width, int height, ByteBuffer rgbBuf, int byteOffset);
    public static native long nativeBitmapFromPcm(float[] pcmData, int nSamples);
    public static native long nativeBitmapFromPcmDirectBuffer(ByteBuffer pcmBuf, int byteOffset, int nSamples);
    public static native long nativeBitmapLoadFile(long mctxHandle, byte[] pathUtf8, boolean placeholder);
    public static native long nativeBitmapLoadBuffer(long mctxHandle, byte[] buffer, int offset, int size, boolean placeholder);
    public static native void nativeBitmapFree(long bitmapHandle);
    public static native boolean nativeBitmapSetMergeable(long bitmapHandle, boolean mergeable);

    public static native long nativeVideoLoadFile(long mctxHandle, byte[] pathUtf8, float fpsTarget, long timestampIntervalMs);
    public static native long nativeVideoLoadBuffer(
        long mctxHandle,
        byte[] buffer,
        int offset,
        int size,
        float fpsTarget,
        long timestampIntervalMs
    );
    public static native void nativeVideoFree(long videoHandle);
    public static native int nativeVideoReadNext(long videoHandle, long[] outBitmapHandle, byte[][] outTextBytes);

    public static native long nativeInputChunksInit();
    public static native void nativeInputChunksFree(long chunksHandle);
    public static native int nativeMultimodalTokenize(
        long mctxHandle,
        long chunksHandle,
        byte[] utf8Text,
        boolean addBos,
        long[] bitmapHandles,
        int nBitmaps
    );
    public static native int nativeEvalMultimodalChunks(
        long mctxHandle,
        long ctxHandle,
        long chunksHandle,
        int nPast,
        int seqId,
        int nBatch,
        boolean logitsLast,
        int[] outNewNPast
    );

    private static void registerErrorResolver() {
        ArgusNativeException.registerErrorResolver(new ArgusNativeException.ErrorResolver() {
            @Override
            public int lastErrorCode() {
                return nativeLastErrorCode();
            }

            @Override
            public String lastErrorMessage() {
                String msg = decodeUtf8(nativeLastErrorMessage());
                return (msg != null && !msg.isEmpty()) ? msg : "Unknown native error";
            }

            @Override
            public void clearError() {
                nativeClearError();
            }
        });
    }
}
