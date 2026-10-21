package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import cc.projectargus.libargus.internal.ArgusSequenceSlotPool;
import cc.projectargus.libargus.internal.ArgusTextGenerator;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Predicate;

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
    private final ArgusSequenceSlotPool slotPool;

    private ArgusContext(long ctxHandle, ArgusModel modelRef, ArgusModel draftModelRef, ArgusContextConfig config) {
        super(ctxHandle);
        this.modelRef = Objects.requireNonNull(modelRef, "modelRef");
        this.draftModelRef = draftModelRef;
        this.slotPool = new ArgusSequenceSlotPool(getSeqMax(), (config != null) ? config.cloneSlots() : 0);
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
        ArgusJniValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
        ArgusJniValidation.checkNonNegative(startPos, "startPos");
        ArgusJniValidation.checkNonNegative(seqId, "seqId");
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
        ArgusJniValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
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
        return slotPool.forkSlot(srcSeqId, getSeqMax(), this::copySequenceSlot);
    }

    public void freeSlot(int seqId) {
        slotPool.freeSlot(seqId, slot -> clearCacheSlot(slot, 0, -1));
    }

    public int getAvailableSlotCount() {
        return slotPool.getAvailableSlotCount();
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
        ArgusJniValidation.checkPositive(nTokens, "nTokens");
        ArgusJniValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
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
        return ArgusTextGenerator.extractEmbeddings(createGenerationOps(null), text);
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
        return ArgusTextGenerator.generate(
            createGenerationOps(abortFlag),
            prompt,
            maxTokens,
            config,
            onPiece,
            abortFlag != null ? abortFlag::isAborted : null
        );
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming
     * with an early-stopping predicate callback. Returning {@code false} from {@code shouldContinue}
     * stops generation immediately.
     */
    public String generateUntil(String prompt, int maxTokens, ArgusSamplerConfig config, Predicate<String> shouldContinue) {
        return generateUntil(prompt, maxTokens, config, shouldContinue, null);
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming,
     * an early-stopping predicate callback, and optional cooperative cancellation.
     */
    public String generateUntil(String prompt, int maxTokens, ArgusSamplerConfig config,
                                Predicate<String> shouldContinue, ArgusAbortFlag abortFlag) {
        return ArgusTextGenerator.generateUntil(
            createGenerationOps(abortFlag),
            prompt,
            maxTokens,
            config,
            shouldContinue,
            abortFlag != null ? abortFlag::isAborted : null
        );
    }

    private ArgusTextGenerator.ContextOps createGenerationOps(ArgusAbortFlag abortFlag) {
        return new ArgusTextGenerator.ContextOps() {
            @Override
            public boolean clearCacheSlot(int seqId, int p0, int p1) {
                return ArgusContext.this.clearCacheSlot(seqId, p0, p1);
            }

            @Override
            public void resetSampler(int seqId) {
                ArgusContext.this.resetSampler(seqId);
            }

            @Override
            public int[] tokenize(String text, boolean addBos) {
                return ArgusContext.this.tokenize(text, addBos);
            }

            @Override
            public int decodeBatch(int[] tokens, int nTokens, int startPos, boolean requestLogits) {
                return ArgusContext.this.decodeBatch(tokens, nTokens, startPos, 0, requestLogits, abortFlag);
            }

            @Override
            public int primeSampler(int seqId, int[] tokens, int nTokens) {
                return ArgusContext.this.primeSampler(seqId, tokens, nTokens);
            }

            @Override
            public int sampleToken(int seqId, ArgusSamplerConfig config) {
                return ArgusContext.this.sampleToken(seqId, config);
            }

            @Override
            public boolean vocabIsEog(int token) {
                return ArgusContext.this.vocabIsEog(token);
            }

            @Override
            public byte[] tokenToPieceBytes(int token) {
                return ArgusContext.this.tokenToPieceBytes(token);
            }

            @Override
            public int nEmbd() {
                return ArgusContext.this.nEmbd();
            }

            @Override
            public int evalEmbeddings(int[] tokens, int nTokens, int seqId, float[] outEmbeddings) {
                return ArgusContext.this.evalEmbeddings(tokens, nTokens, seqId, outEmbeddings);
            }
        };
    }
}
