package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import cc.projectargus.libargus.internal.ArgusUtf8StreamDecoder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Represents an active text inference, embedding, multimodal, and speech generation session context
 * via the JDK 17+ / Android-compatible JNI bridge.
 * Extends {@link ArgusJniResource} for thread-safe handle leasing and idempotent lifecycle management.
 */
public final class ArgusContext extends ArgusJniResource {
    public static final int STATUS_SUCCESS = 0;
    public static final int STATUS_ERROR = -1;
    public static final int STATUS_CANCELLED = -2;
    public static final int STATUS_ROLLBACK_FAILED = -3;

    private final ArgusModel modelRef;
    private final ArgusModel draftModelRef;
    private final BitSet leasedSlots = new BitSet();
    private final BitSet releasingSlots = new BitSet();
    private final Object slotPoolLock = new Object();
    private final int cloneSlotStart;
    private final int cloneSlotCount;

    private ArgusContext(long ctxHandle, ArgusModel modelRef, ArgusModel draftModelRef, ArgusContextConfig config) {
        super(ctxHandle);
        this.modelRef = Objects.requireNonNull(modelRef, "modelRef");
        this.draftModelRef = draftModelRef;
        int maxSlots = getSeqMax();
        int configuredClone = (config != null) ? config.cloneSlots() : 0;
        this.cloneSlotCount = Math.max(0, Math.min(configuredClone, maxSlots));
        this.cloneSlotStart = Math.max(0, maxSlots - this.cloneSlotCount);
        this.leasedSlots.set(0); // Slot 0 is reserved as root sequence slot
    }

    @Override
    protected String resourceName() {
        return "ArgusContext";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeContextFree(oldHandle);
    }

    /**
     * Initializes an execution context on the specified model weights.
     */
    public static ArgusContext init(ArgusModel model, ArgusContextConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");

        try (var modelLease = model.lease()) {
            ArgusJniResource.Lease draftLease = (config.draftModel() != null) ? config.draftModel().lease() : null;
            try {
                long draftHandle = (draftLease != null) ? draftLease.handle() : 0L;
                long ctxPtr = ArgusJni.nativeContextInit(
                    modelLease.handle(),
                    draftHandle,
                    config.contextLength(),
                    config.cpuThreads(),
                    config.typeK(),
                    config.typeV(),
                    config.specDraftNMax(),
                    config.uBatch(),
                    config.seqMax(),
                    config.enableDraftMtp(),
                    config.embeddings(),
                    config.kvUnified()
                );
                if (ctxPtr == 0L) {
                    ArgusNativeException.throwLastError("argus_context_init");
                }
                try {
                    ArgusContext contextInstance = new ArgusContext(ctxPtr, model, config.draftModel(), config);
                    if (config.causalAttn() != null) {
                        contextInstance.setCausalAttn(config.causalAttn());
                    }
                    return contextInstance;
                } catch (Throwable t) {
                    try {
                        ArgusJni.nativeContextFree(ctxPtr);
                    } catch (Throwable suppressed) {
                        t.addSuppressed(suppressed);
                    }
                    throw t;
                }
            } finally {
                if (draftLease != null) {
                    draftLease.close();
                }
            }
        }
    }

    /**
     * Converts a Java string into vocabulary token IDs into a caller-supplied int array using the context's retained model.
     */
    public int tokenize(String text, int[] outTokens, int maxTokens, boolean addBos) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(outTokens, "outTokens");
        ArgusJniValidation.checkPositive(maxTokens, "maxTokens");
        if (maxTokens > outTokens.length) {
            throw new IllegalArgumentException("maxTokens (" + maxTokens + ") exceeds outTokens.length (" + outTokens.length + ")");
        }
        byte[] utf8 = ArgusJniValidation.toUtf8(text, "text");
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int res = ArgusJni.nativeTokenize(modelH, utf8, outTokens, maxTokens, addBos);
            if (res == Integer.MIN_VALUE) {
                throw new ArgusNativeException(
                    ArgusNativeException.ERROR_INTERNAL,
                    "Tokenization overflow: result size exceeds int32 limit"
                );
            }
            if (res < 0) {
                int lastErr = ArgusJni.nativeLastErrorCode();
                if (lastErr != 0) {
                    ArgusNativeException.checkStatus(res, "argus_tokenize_n");
                }
            }
            return res;
        });
    }

    /**
     * Converts a Java string into vocabulary token IDs as a primitive Java int array using the context's retained model.
     */
    public int[] tokenize(String text, boolean addBos) {
        Objects.requireNonNull(text, "text");
        byte[] utf8 = ArgusJniValidation.toUtf8(text, "text");
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int initialCap = Math.max(64, utf8.length);
            int[] buf = new int[initialCap];
            int nTokens = ArgusJni.nativeTokenize(modelH, utf8, buf, buf.length, addBos);
            if (nTokens == Integer.MIN_VALUE) {
                throw new ArgusNativeException(
                    ArgusNativeException.ERROR_INTERNAL,
                    "Tokenization overflow: result size exceeds int32 limit"
                );
            }
            if (nTokens < 0) {
                int lastErr = ArgusJni.nativeLastErrorCode();
                if (lastErr != 0) {
                    ArgusNativeException.checkStatus(nTokens, "argus_tokenize_n");
                }
                int required = -nTokens;
                if (required <= 0 || required > 10_000_000) {
                    throw new IllegalArgumentException("Token count exceeds safety limit: " + required);
                }
                int[] exactBuf = new int[required];
                int retry = ArgusJni.nativeTokenize(modelH, utf8, exactBuf, required, addBos);
                if (retry < 0) {
                    ArgusNativeException.checkStatus(retry, "argus_tokenize_n");
                }
                return retry == required ? exactBuf : Arrays.copyOf(exactBuf, retry);
            }
            return Arrays.copyOf(buf, nTokens);
        });
    }

    /**
     * Converts a single token ID into its raw UTF-8 byte sequence using the context's retained model.
     */
    public byte[] tokenToPieceBytes(int token) {
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            byte[] bytes = ArgusJni.nativeTokenToPiece(modelH, token);
            return bytes != null ? bytes : new byte[0];
        });
    }

    /**
     * Converts a single token ID to its text representation (piece) using the context's retained model.
     */
    public String tokenToPiece(int token) {
        byte[] bytes = tokenToPieceBytes(token);
        if (bytes.length == 0) {
            return "";
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Converts a single token ID into raw UTF-8 bytes written directly into a caller-supplied byte array.
     */
    public int tokenToPieceInto(int token, byte[] dst, int offset, int maxLen) {
        Objects.requireNonNull(dst, "dst");
        if (offset < 0 || maxLen <= 0 || (long) offset + (long) maxLen > dst.length) {
            throw new IllegalArgumentException(
                "Invalid slice bounds: offset=" + offset + ", maxLen=" + maxLen + ", dst.length=" + dst.length
            );
        }
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            return ArgusJni.nativeTokenToPieceInto(modelH, token, dst, offset, maxLen);
        });
    }

    /**
     * Checks if the given token ID is an End-Of-Generation (EOG) token using the context's retained model.
     */
    public boolean vocabIsEog(int token) {
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            return ArgusJni.nativeVocabIsEog(modelH, token);
        });
    }

    /**
     * Synchronously decodes a batch of tokens from a primitive int array.
     */
    public int decodeBatch(int[] tokens, int nTokens, int startPos, int seqId, boolean requestLogits) {
        return decodeBatch(tokens, 0, nTokens, startPos, seqId, requestLogits, null);
    }

    /**
     * Synchronously decodes a batch of tokens from a primitive int array with optional early cancellation.
     */
    public int decodeBatch(int[] tokens, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        return decodeBatch(tokens, 0, nTokens, startPos, seqId, requestLogits, abortFlag);
    }

    /**
     * Synchronously decodes a slice of tokens from a primitive int array with optional early cancellation.
     */
    public int decodeBatch(int[] tokens, int offset, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(tokens, "tokens");
        ArgusJniValidation.checkNonNegative(offset, "offset");
        ArgusJniValidation.checkNonNegative(nTokens, "nTokens");
        ArgusJniValidation.checkNonNegative(startPos, "startPos");
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
        if ((long) offset + (long) nTokens > tokens.length) {
            throw new IllegalArgumentException(
                "Token slice [offset=" + offset + ", nTokens=" + nTokens + ") exceeds tokens.length (" + tokens.length + ")"
            );
        }
        if (nTokens == 0) {
            return 0;
        }
        return withHandle(ctxH -> {
            if (abortFlag != null) {
                try (var abortLease = abortFlag.lease()) {
                    return ArgusJni.nativeDecodeTokensArray(ctxH, tokens, offset, nTokens, startPos, seqId, requestLogits, abortLease.handle());
                }
            }
            return ArgusJni.nativeDecodeTokensArray(ctxH, tokens, offset, nTokens, startPos, seqId, requestLogits, 0L);
        });
    }

    /**
     * Synchronously decodes a batch of tokens from a direct NIO ByteBuffer (containing 4-byte native-endian ints).
     */
    public int decodeBatchDirect(ByteBuffer tokensDirectBuf, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(tokensDirectBuf, "tokensDirectBuf");
        ArgusJniValidation.checkNonNegative(nTokens, "nTokens");
        ArgusJniValidation.checkNonNegative(startPos, "startPos");
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
        if (nTokens == 0) {
            return 0;
        }
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(nTokens, Integer.BYTES, "tokensDirectBuf");
        ArgusJniValidation.checkDirectBuffer(tokensDirectBuf, requiredBytes, "tokensDirectBuf");
        int byteOffset = tokensDirectBuf.position();
        if ((byteOffset & 3) != 0) {
            throw new IllegalArgumentException("tokensDirectBuf.position() (" + byteOffset + ") must be 4-byte aligned");
        }
        return withHandle(ctxH -> {
            if (abortFlag != null) {
                try (var abortLease = abortFlag.lease()) {
                    return ArgusJni.nativeDecodeTokensDirectBuffer(ctxH, tokensDirectBuf, byteOffset, nTokens, startPos, seqId, requestLogits, abortLease.handle());
                }
            }
            return ArgusJni.nativeDecodeTokensDirectBuffer(ctxH, tokensDirectBuf, byteOffset, nTokens, startPos, seqId, requestLogits, 0L);
        });
    }

    /**
     * Evaluates tokenized multimodal chunks (combining text and media projectors).
     */
    public int evalMultimodalChunks(ArgusMultimodalContext mctx, ArgusInputChunks chunks, int nPast, int seqId, int nBatch, boolean logitsLast) {
        Objects.requireNonNull(mctx, "mctx");
        Objects.requireNonNull(chunks, "chunks");
        ArgusJniValidation.checkNonNegative(nPast, "nPast");
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
        ArgusJniValidation.checkPositive(nBatch, "nBatch");

        return withHandle(ctxH -> {
            try (var mctxLease = mctx.lease();
                 var chunksLease = chunks.lease()) {
                int[] outNewNPast = new int[1];
                int res = ArgusJni.nativeEvalMultimodalChunks(
                    mctxLease.handle(),
                    ctxH,
                    chunksLease.handle(),
                    nPast,
                    seqId,
                    nBatch,
                    logitsLast,
                    outNewNPast
                );
                if (res != 0) {
                    ArgusNativeException.checkStatus(res, "argus_eval_multimodal_chunks");
                }
                return outNewNPast[0];
            }
        });
    }

    public boolean hasDraftContext() {
        return withHandle(ArgusJni::nativeContextHasDraft);
    }

    public int sampleToken(int seqId, float temperature, float repeatPenalty) {
        return withHandle(ctxH -> ArgusJni.nativeSampleToken(ctxH, seqId, temperature, repeatPenalty));
    }

    public int sampleToken(int seqId, ArgusSamplerConfig config) {
        return sampleTokenWithBias(seqId, config, null, null, 0);
    }

    public int sampleTokenWithBias(int seqId, float temperature, float repeatPenalty,
                                   int[] biasTokens, float[] biasValues, int biasCount) {
        ArgusJniValidation.checkNonNegative(biasCount, "biasCount");
        if (biasCount > 0) {
            Objects.requireNonNull(biasTokens, "biasTokens");
            Objects.requireNonNull(biasValues, "biasValues");
            if (biasCount > biasTokens.length || biasCount > biasValues.length) {
                throw new IllegalArgumentException("biasCount exceeds bias array length");
            }
        }
        return withHandle(ctxH -> ArgusJni.nativeSampleTokenWithBias(
            ctxH, seqId, temperature, repeatPenalty, biasTokens, biasValues, biasCount
        ));
    }

    public int sampleTokenWithBias(int seqId, ArgusSamplerConfig config,
                                   int[] biasTokens, float[] biasValues, int biasCount) {
        Objects.requireNonNull(config, "config");
        ArgusJniValidation.checkNonNegative(biasCount, "biasCount");
        if (biasCount > 0) {
            Objects.requireNonNull(biasTokens, "biasTokens");
            Objects.requireNonNull(biasValues, "biasValues");
            if (biasCount > biasTokens.length || biasCount > biasValues.length) {
                throw new IllegalArgumentException("biasCount exceeds bias array length");
            }
        }
        return withHandle(ctxH -> ArgusJni.nativeSampleTokenExt(
            ctxH,
            seqId,
            config.temperature(),
            config.repeatPenalty(),
            config.repeatLastN(),
            config.frequencyPenalty(),
            config.presencePenalty(),
            config.topP(),
            config.minP(),
            config.topK(),
            config.dryMultiplier(),
            config.dryBase(),
            config.dryAllowedLength(),
            config.dryPenaltyLastN(),
            config.seed(),
            biasTokens,
            biasValues,
            biasCount
        ));
    }

    public void resetSampler(int seqId) {
        withHandleVoid(ctxH -> {
            int res = ArgusJni.nativeSamplerReset(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_reset");
            }
        });
    }

    public int primeSampler(int seqId, int[] tokens, int nTokens) {
        return primeSampler(seqId, tokens, 0, nTokens);
    }

    public int primeSampler(int seqId, int[] tokens, int offset, int nTokens) {
        Objects.requireNonNull(tokens, "tokens");
        ArgusJniValidation.checkNonNegative(offset, "offset");
        ArgusJniValidation.checkNonNegative(nTokens, "nTokens");
        if ((long) offset + (long) nTokens > tokens.length) {
            throw new IllegalArgumentException(
                "Token slice [offset=" + offset + ", nTokens=" + nTokens + ") exceeds tokens.length (" + tokens.length + ")"
            );
        }
        if (nTokens == 0) {
            return 0;
        }
        return withHandle(ctxH -> {
            int res = ArgusJni.nativeSamplerPrime(ctxH, seqId, tokens, offset, nTokens);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_prime");
            }
            return res;
        });
    }

    public void truncateSampler(int seqId, int newLength) {
        ArgusJniValidation.checkNonNegative(newLength, "newLength");
        withHandleVoid(ctxH -> {
            int res = ArgusJni.nativeSamplerTruncate(ctxH, seqId, newLength);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_truncate");
            }
        });
    }

    public int getSamplerHistoryCount(int seqId) {
        return withHandle(ctxH -> {
            int res = ArgusJni.nativeSamplerGetHistoryCount(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_get_history_count");
            }
            return res;
        });
    }

    public boolean hasSamplerPending(int seqId) {
        return withHandle(ctxH -> {
            int res = ArgusJni.nativeSamplerHasPending(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_has_pending");
            }
            return res == 1;
        });
    }

    public boolean discardPendingSample(int seqId) {
        return withHandle(ctxH -> {
            int res = ArgusJni.nativeSamplerDiscardPending(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_discard_pending");
            }
            return res == 1;
        });
    }

    public boolean clearCacheSlot(int seqId, int p0, int p1) {
        return withHandle(ctxH -> ArgusJni.nativeKvCacheClearSlot(ctxH, seqId, p0, p1));
    }

    public boolean copySequenceSlot(int srcSeqId, int dstSeqId, int p0, int p1) {
        ArgusJniValidation.checkNonNegative(srcSeqId, "srcSeqId");
        ArgusJniValidation.checkNonNegative(dstSeqId, "dstSeqId");
        return withHandle(ctxH -> ArgusJni.nativeKvCacheSeqCp(ctxH, srcSeqId, dstSeqId, p0, p1));
    }

    public boolean copySequenceSlot(int srcSeqId, int dstSeqId) {
        return copySequenceSlot(srcSeqId, dstSeqId, 0, -1);
    }

    public boolean canShift() {
        return withHandle(ArgusJni::nativeKvCacheCanShift);
    }

    public boolean shiftCacheSlot(int seqId, int p0, int p1, int delta) {
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
        return withHandle(ctxH -> ArgusJni.nativeKvCacheSeqAdd(ctxH, seqId, p0, p1, delta));
    }

    public int getSeqMax() {
        return withHandle(ArgusJni::nativeContextGetSeqMax);
    }

    public int forkSlot(int srcSeqId) {
        ArgusJniValidation.checkNonNegative(srcSeqId, "srcSeqId");
        int maxSlots = getSeqMax();
        if (srcSeqId >= maxSlots) {
            throw new IllegalArgumentException("srcSeqId " + srcSeqId + " exceeds max sequence slots " + maxSlots);
        }
        if (cloneSlotCount == 0) {
            throw new IllegalStateException("Clone pool unconfigured (cloneSlots = 0); increase cloneSlots in ArgusContextConfig");
        }
        int leasedSlot;
        synchronized (slotPoolLock) {
            leasedSlot = -1;
            for (int i = cloneSlotStart; i < cloneSlotStart + cloneSlotCount; i++) {
                if (!leasedSlots.get(i) && !releasingSlots.get(i)) {
                    leasedSlot = i;
                    leasedSlots.set(i);
                    break;
                }
            }
        }
        if (leasedSlot < 0) {
            throw new IllegalStateException("All " + cloneSlotCount + " clone worker slots are currently leased; increase cloneSlots in ArgusContextConfig");
        }
        boolean success = false;
        try {
            boolean ok = copySequenceSlot(srcSeqId, leasedSlot);
            if (!ok) {
                ArgusNativeException.throwLastError("forkSlot");
            }
            success = true;
            return leasedSlot;
        } finally {
            if (!success) {
                synchronized (slotPoolLock) {
                    leasedSlots.clear(leasedSlot);
                }
            }
        }
    }

    public void freeSlot(int seqId) {
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
        if (cloneSlotCount == 0 || seqId < cloneSlotStart || seqId >= cloneSlotStart + cloneSlotCount) {
            throw new IllegalArgumentException(
                "seqId " + seqId + " does not belong to the configured clone pool ["
                    + cloneSlotStart + ", " + (cloneSlotStart + cloneSlotCount) + ")"
            );
        }
        synchronized (slotPoolLock) {
            if (!leasedSlots.get(seqId) || releasingSlots.get(seqId)) {
                throw new IllegalStateException("Slot " + seqId + " is not currently leased or is already being released");
            }
            releasingSlots.set(seqId);
        }
        boolean ok = false;
        try {
            ok = clearCacheSlot(seqId, 0, -1);
            if (!ok) {
                ArgusNativeException.throwLastError("freeSlot");
            }
        } finally {
            synchronized (slotPoolLock) {
                if (ok) {
                    leasedSlots.clear(seqId);
                    releasingSlots.clear(seqId);
                } else {
                    releasingSlots.clear(seqId);
                }
            }
        }
    }

    public int getAvailableSlotCount() {
        if (cloneSlotCount == 0) {
            return 0;
        }
        synchronized (slotPoolLock) {
            int inUse = 0;
            for (int i = cloneSlotStart; i < cloneSlotStart + cloneSlotCount; i++) {
                if (leasedSlots.get(i) || releasingSlots.get(i)) inUse++;
            }
            return Math.max(0, cloneSlotCount - inUse);
        }
    }

    public boolean isHybrid() {
        return modelRef.isHybrid();
    }

    public boolean isRecurrent() {
        return modelRef.isRecurrent();
    }

    public boolean isDiffusion() {
        return modelRef.isDiffusion();
    }

    public int getSeqPosMax(int seqId) {
        return withHandle(ctxH -> ArgusJni.nativeKvCacheSeqPosMax(ctxH, seqId));
    }

    public int getSeqPosMin(int seqId) {
        return withHandle(ctxH -> ArgusJni.nativeKvCacheSeqPosMin(ctxH, seqId));
    }

    /**
     * Synthesizes speech from a Java String into a caller-provided float array.
     */
    public int synthesizeSpeech(ArgusModel wavtokenizerModel, String text, int voiceSeed, float[] outPcm, int maxSamples) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(outPcm, "outPcm");
        ArgusJniValidation.checkNonNegative(maxSamples, "maxSamples");
        if (maxSamples > outPcm.length) {
            throw new IllegalArgumentException("maxSamples (" + maxSamples + ") exceeds outPcm.length (" + outPcm.length + ")");
        }
        if (maxSamples == 0) {
            return 0;
        }
        byte[] textUtf8 = ArgusJniValidation.toUtf8(text, "text");
        return withHandle(ctxH -> {
            ArgusJniResource.Lease wavLease = (wavtokenizerModel != null) ? wavtokenizerModel.lease() : null;
            try {
                long wavHandle = (wavLease != null) ? wavLease.handle() : 0L;
                int res = ArgusJni.nativeSynthesizeSpeech(ctxH, wavHandle, textUtf8, voiceSeed, outPcm, maxSamples);
                if (res < 0) {
                    ArgusNativeException.checkStatus(res, "argus_synthesize_speech_n");
                }
                return res;
            } finally {
                if (wavLease != null) {
                    wavLease.close();
                }
            }
        });
    }

    /**
     * Synthesizes speech from a Java String and returns a newly allocated float array of the exact sample count.
     */
    public float[] synthesizeSpeech(ArgusModel wavtokenizerModel, String text, int voiceSeed, int maxSamples) {
        Objects.requireNonNull(text, "text");
        ArgusJniValidation.checkPositive(maxSamples, "maxSamples");
        float[] buf = new float[maxSamples];
        int written = synthesizeSpeech(wavtokenizerModel, text, voiceSeed, buf, maxSamples);
        if (written <= 0) {
            return new float[0];
        }
        return written == maxSamples ? buf : Arrays.copyOf(buf, written);
    }

    public int getEmbeddings(int seqId, float[] outEmbeddings, int maxFloats) {
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        ArgusJniValidation.checkNonNegative(maxFloats, "maxFloats");
        if (maxFloats > outEmbeddings.length) {
            throw new IllegalArgumentException("maxFloats (" + maxFloats + ") exceeds outEmbeddings.length (" + outEmbeddings.length + ")");
        }
        if (maxFloats == 0) {
            return 0;
        }
        return withHandle(ctxH -> {
            int written = ArgusJni.nativeGetEmbeddings(ctxH, seqId, outEmbeddings, maxFloats);
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_get_embeddings");
            }
            return written;
        });
    }

    public int getEmbeddings(float[] outEmbeddings, int maxFloats) {
        return getEmbeddings(0, outEmbeddings, maxFloats);
    }

    public float[] getEmbeddings(int seqId) {
        int dim = nEmbd();
        float[] out = new float[dim];
        int written = getEmbeddings(seqId, out, dim);
        if (written != dim) {
            throw new IllegalStateException("Incomplete embedding retrieval: expected " + dim + " floats, got " + written);
        }
        return out;
    }

    public float[] getEmbeddings() {
        return getEmbeddings(0);
    }

    /**
     * Retrieves the embedding vector for a specific sequence ID directly into a writable direct NIO ByteBuffer.
     */
    public int getEmbeddingsDirect(int seqId, ByteBuffer outEmbeddingsDirectBuf, int maxFloats) {
        Objects.requireNonNull(outEmbeddingsDirectBuf, "outEmbeddingsDirectBuf");
        ArgusJniValidation.checkNonNegative(maxFloats, "maxFloats");
        if (maxFloats == 0) {
            return 0;
        }
        long requiredBytes = ArgusJniValidation.multiplyExactBytes(maxFloats, Float.BYTES, "outEmbeddingsDirectBuf");
        ArgusJniValidation.checkDirectWritableBuffer(outEmbeddingsDirectBuf, requiredBytes, "outEmbeddingsDirectBuf");
        int byteOffset = outEmbeddingsDirectBuf.position();
        if ((byteOffset & 3) != 0) {
            throw new IllegalArgumentException("outEmbeddingsDirectBuf.position() (" + byteOffset + ") must be 4-byte aligned");
        }
        return withHandle(ctxH -> {
            int written = ArgusJni.nativeGetEmbeddingsDirectBuffer(ctxH, seqId, outEmbeddingsDirectBuf, byteOffset, maxFloats);
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_get_embeddings");
            }
            return written;
        });
    }

    public void setNThreads(int nThreads, int nThreadsBatch) {
        ArgusJniValidation.checkPositive(nThreads, "nThreads");
        ArgusJniValidation.checkPositive(nThreadsBatch, "nThreadsBatch");
        withHandleVoid(ctxH -> ArgusJni.nativeSetNThreads(ctxH, nThreads, nThreadsBatch));
    }

    public int getNThreads() {
        return withHandle(ArgusJni::nativeGetNThreads);
    }

    public int getNThreadsBatch() {
        return withHandle(ArgusJni::nativeGetNThreadsBatch);
    }

    public ArgusPerfTimings getPerfTimings() {
        return withHandle(ctxH -> {
            double[] doubles = new double[4];
            int[] ints = new int[5];
            boolean ok = ArgusJni.nativeContextGetPerf(ctxH, doubles, ints);
            if (!ok) {
                ArgusNativeException.throwLastError("argus_context_get_perf");
            }
            return new ArgusPerfTimings(
                doubles[0], doubles[1], doubles[2], doubles[3],
                ints[0], ints[1], ints[2], ints[3], ints[4]
            );
        });
    }

    public void resetPerfTimings() {
        withHandleVoid(ArgusJni::nativeContextResetPerf);
    }

    public void setCausalAttn(boolean causal) {
        withHandleVoid(ctxH -> {
            if (!ArgusJni.nativeContextSetCausalAttn(ctxH, causal)) {
                ArgusNativeException.throwLastError("argus_context_set_causal_attn");
            }
        });
    }

    public boolean isCausalAttn() {
        return withHandle(ctxH -> {
            int res = ArgusJni.nativeContextGetCausalAttn(ctxH);
            if (res < 0) {
                ArgusNativeException.throwLastError("argus_context_get_causal_attn");
            }
            return res == 1;
        });
    }

    public int nEmbd() {
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = ArgusJni.nativeModelNEmbd(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd");
            }
            return dim;
        });
    }

    public int nEmbdOut() {
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = ArgusJni.nativeModelNEmbdOut(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd_out");
            }
            return dim;
        });
    }

    public int nEmbdInp() {
        return withHandle(ctxH -> {
            long modelH = ArgusJni.nativeContextGetModel(ctxH);
            if (modelH == 0L) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = ArgusJni.nativeModelNEmbdInp(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd_inp");
            }
            return dim;
        });
    }

    public int evalEmbeddings(int[] tokens, int nTokens, float[] outEmbeddings) {
        return evalEmbeddings(tokens, 0, nTokens, 0, outEmbeddings);
    }

    public int evalEmbeddings(int[] tokens, int nTokens, int seqId, float[] outEmbeddings) {
        return evalEmbeddings(tokens, 0, nTokens, seqId, outEmbeddings);
    }

    public int evalEmbeddings(int[] tokens, int offset, int nTokens, int seqId, float[] outEmbeddings) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        ArgusJniValidation.checkNonNegative(offset, "offset");
        if (nTokens <= 0 || (long) offset + (long) nTokens > tokens.length) {
            throw new IllegalArgumentException(
                "Invalid token slice [offset=" + offset + ", nTokens=" + nTokens + ") for tokens.length=" + tokens.length
            );
        }
        int requiredDim = nEmbd();
        if (outEmbeddings.length < requiredDim) {
            throw new IllegalArgumentException("outEmbeddings capacity (" + outEmbeddings.length +
                " floats) is insufficient for model embedding dimension (" + requiredDim + " floats)");
        }
        return withHandle(ctxH -> {
            int written = ArgusJni.nativeEvalEmbeddings(ctxH, tokens, offset, nTokens, seqId, outEmbeddings, outEmbeddings.length);
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_eval_embeddings");
            }
            if (written != requiredDim) {
                throw new IllegalStateException("Incomplete embedding extraction: expected " + requiredDim + " floats, got " + written);
            }
            return written;
        });
    }

    public int evalEmbeddingsDirect(ByteBuffer tokensDirectBuf, int nTokens, int seqId, ByteBuffer outEmbeddingsDirectBuf) {
        Objects.requireNonNull(tokensDirectBuf, "tokensDirectBuf");
        Objects.requireNonNull(outEmbeddingsDirectBuf, "outEmbeddingsDirectBuf");
        ArgusJniValidation.checkPositive(nTokens, "nTokens");
        int requiredDim = nEmbd();
        long tokBytes = ArgusJniValidation.multiplyExactBytes(nTokens, Integer.BYTES, "tokensDirectBuf");
        long outBytes = ArgusJniValidation.multiplyExactBytes(requiredDim, Float.BYTES, "outEmbeddingsDirectBuf");
        ArgusJniValidation.checkDirectBuffer(tokensDirectBuf, tokBytes, "tokensDirectBuf");
        ArgusJniValidation.checkDirectWritableBuffer(outEmbeddingsDirectBuf, outBytes, "outEmbeddingsDirectBuf");
        int tokOffset = tokensDirectBuf.position();
        int outOffset = outEmbeddingsDirectBuf.position();
        if ((tokOffset & 3) != 0 || (outOffset & 3) != 0) {
            throw new IllegalArgumentException("Direct buffer positions must be 4-byte aligned");
        }
        int maxFloats = outEmbeddingsDirectBuf.remaining() / Float.BYTES;
        return withHandle(ctxH -> {
            int written = ArgusJni.nativeEvalEmbeddingsDirectBuffer(
                ctxH, tokensDirectBuf, tokOffset, nTokens, seqId, outEmbeddingsDirectBuf, outOffset, maxFloats
            );
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_eval_embeddings");
            }
            if (written != requiredDim) {
                throw new IllegalStateException("Incomplete embedding extraction: expected " + requiredDim + " floats, got " + written);
            }
            return written;
        });
    }

    public ArgusModel getModel() {
        return modelRef;
    }

    public float[] extractEmbeddings(String text) {
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) {
            throw new IllegalArgumentException("Cannot extract embeddings for empty text string");
        }
        int[] tokens = tokenize(text, true);
        if (tokens.length == 0) {
            throw new IllegalStateException("Tokenization produced 0 tokens for non-empty text");
        }
        int dim = nEmbd();
        float[] result = new float[dim];
        evalEmbeddings(tokens, tokens.length, 0, result);
        return result;
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming.
     */
    public String generate(String prompt, int maxTokens, ArgusSamplerConfig config, Consumer<String> onPiece) {
        return generate(prompt, maxTokens, config, onPiece, null);
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming
     * and optional cooperative cancellation.
     */
    public String generate(String prompt, int maxTokens, ArgusSamplerConfig config,
                           Consumer<String> onPiece, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(config, "config");
        ArgusJniValidation.checkNonNegative(maxTokens, "maxTokens");
        if (maxTokens == 0) {
            return "";
        }

        clearCacheSlot(0, 0, -1);
        resetSampler(0);

        int[] promptTokens = tokenize(prompt, true);
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("Prompt tokenized to 0 tokens");
        }

        int prefillStatus = decodeBatch(promptTokens, promptTokens.length, 0, 0, true, abortFlag);
        if (prefillStatus == STATUS_CANCELLED) {
            return "";
        }
        if (prefillStatus != 0) {
            ArgusNativeException.checkStatus(prefillStatus, "argus_decode_tokens");
        }
        primeSampler(0, promptTokens, promptTokens.length);

        ArgusUtf8StreamDecoder decoder = new ArgusUtf8StreamDecoder(true);
        int[] singleToken = new int[1];
        int currentPos = promptTokens.length;

        for (int step = 0; step < maxTokens; step++) {
            if (abortFlag != null && abortFlag.isAborted()) {
                break;
            }
            int nextToken = sampleToken(0, config);
            if (nextToken < 0) {
                ArgusNativeException.checkStatus(nextToken, "argus_sample_token_ext");
            }
            if (vocabIsEog(nextToken)) {
                break;
            }

            byte[] pieceBytes = tokenToPieceBytes(nextToken);
            decoder.accept(pieceBytes, 0, pieceBytes.length, onPiece);

            if (step + 1 < maxTokens) {
                singleToken[0] = nextToken;
                int stepStatus = decodeBatch(singleToken, 1, currentPos, 0, true, abortFlag);
                if (stepStatus == STATUS_CANCELLED) {
                    break;
                }
                if (stepStatus != 0) {
                    ArgusNativeException.checkStatus(stepStatus, "argus_decode_tokens");
                }
                currentPos++;
            }
        }

        return decoder.finish(onPiece);
    }
}
