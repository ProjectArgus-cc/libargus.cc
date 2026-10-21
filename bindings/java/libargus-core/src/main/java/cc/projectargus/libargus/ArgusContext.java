package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import cc.projectargus.libargus.internal.ArgusSequenceSlotPool;
import cc.projectargus.libargus.internal.ArgusTextGenerator;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Represents an active text inference and speech generation session context.
 * Owns a private shared arena for concurrent thread safety.
 * Extends {@link ArgusNativeResource} for thread-safe handle leasing and idempotent lifecycle management.
 */
public final class ArgusContext extends ArgusNativeResource {
    public static final int STATUS_SUCCESS = 0;
    public static final int STATUS_ERROR = -1;
    public static final int STATUS_CANCELLED = -2;
    public static final int STATUS_ROLLBACK_FAILED = -3;

    private final ArgusModel modelRef;
    private final ArgusModel draftModelRef;
    private final Arena contextArena;
    private final MemorySegment samplerParamsSeg;
    private final ReentrantLock samplerLock = new ReentrantLock();
    private final ReentrantLock ttsLock = new ReentrantLock();

    private final ArgusSequenceSlotPool slotPool;

    private MemorySegment ttsWorkspace = MemorySegment.NULL;
    private long ttsWorkspaceSizeFloats = 0;

    MemorySegment getTtsWorkspace() {
        return ttsWorkspace;
    }

    long getTtsWorkspaceSizeFloats() {
        return ttsWorkspaceSizeFloats;
    }

    private ArgusContext(MemorySegment ctxPtr, ArgusModel modelRef, ArgusModel draftModelRef, Arena contextArena, ArgusContextConfig config) {
        super(ctxPtr);
        this.modelRef = Objects.requireNonNull(modelRef);
        this.draftModelRef = draftModelRef;
        this.contextArena = Objects.requireNonNull(contextArena);
        this.samplerParamsSeg = contextArena.allocate(ArgusLayouts.SAMPLER_PARAMS);
        this.slotPool = new ArgusSequenceSlotPool(getSeqMax(), (config != null) ? config.cloneSlots() : 0);
    }

    @Override
    protected String resourceName() {
        return "ArgusContext";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_context_free.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        } finally {
            ttsWorkspace = MemorySegment.NULL;
            ttsWorkspaceSizeFloats = 0;
            contextArena.close();
        }
    }

    /**
     * Initializes an execution context on the specified model weights using a private shared arena.
     *
     * @param model  the loaded model weights
     * @param config session parameters (length, threads, cache formats, etc.)
     * @return an active ArgusContext
     */
    public static ArgusContext init(ArgusModel model, ArgusContextConfig config) {
        return init(null, model, config);
    }

    /**
     * Initializes an execution context on the specified model weights.
     * Context creates and owns a private shared arena for its internal unmanaged structs.
     *
     * @param arena  optional caller arena for parameter structures (if null, private shared arena is used)
     * @param model  the loaded model weights
     * @param config session parameters (length, threads, cache formats, etc.)
     * @return an active ArgusContext
     */
    public static ArgusContext init(Arena arena, ArgusModel model, ArgusContextConfig config) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(config);

        Arena privateArena = Arena.ofShared();
        try (var modelLease = model.lease()) {
            ArgusNativeResource.Lease draftLease = null;
            if (config.draftModel() != null) {
                draftLease = config.draftModel().lease();
            }
            try {
                MemorySegment draftModelHandle = (draftLease != null) ? draftLease.handle() : MemorySegment.NULL;
                Arena paramArena = (arena != null) ? arena : privateArena;
                MemorySegment paramsSeg = paramArena.allocate(ArgusLayouts.CONTEXT_PARAMS);

                paramsSeg.set(ValueLayout.ADDRESS, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("draft_model")), 
                    draftModelHandle
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("context_length")), 
                    config.contextLength()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("cpu_threads")), 
                    config.cpuThreads()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("type_k")), 
                    config.typeK()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("type_v")), 
                    config.typeV()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("spec_draft_n_max")), 
                    config.specDraftNMax()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("u_batch")), 
                    config.uBatch()
                );
                paramsSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("n_seq_max")), 
                    config.seqMax()
                );
                paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("enable_draft_mtp")), 
                    config.enableDraftMtp()
                );
                paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("embeddings")), 
                    config.embeddings()
                );
                paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                    ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("kv_unified")), 
                    config.kvUnified()
                );

                MemorySegment ctxPtr = (MemorySegment) ArgusBindings.argus_context_init.invokeExact(modelLease.handle(), paramsSeg);
                if (ctxPtr.equals(MemorySegment.NULL)) {
                    ArgusNativeException.throwLastError("argus_context_init");
                }
                try {
                    ArgusContext contextInstance = new ArgusContext(ctxPtr, model, config.draftModel(), privateArena, config);
                    if (config.causalAttn() != null) {
                        contextInstance.setCausalAttn(config.causalAttn());
                    }
                    return contextInstance;
                } catch (Throwable t) {
                    try {
                        ArgusBindings.argus_context_free.invokeExact(ctxPtr);
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
        } catch (Throwable t) {
            privateArena.close();
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to initialize native context", t);
        }
    }

    /**
     * Converts raw text into vocabulary token IDs.
     * Operates on the execution context's retained model reference, allowing this method
     * to continue functioning even if the original Java ArgusModel wrapper has been closed.
     * This operation is lock-free and read-only.
     */
    public int tokenize(MemorySegment textSeg, MemorySegment outTokensSeg, boolean addBos) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outTokensSeg);
        ArgusValidation.checkReadable(textSeg, 0, "textSeg");
        ArgusValidation.checkWritable(outTokensSeg, ValueLayout.JAVA_INT.byteSize(), "outTokensSeg");
        long maxTokens = outTokensSeg.byteSize() / ValueLayout.JAVA_INT.byteSize();
        long textLen = textSeg.byteSize();

        MemorySegment ctxH = acquireReadLease();
        try {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int res = (int) ArgusBindings.argus_tokenize_n.invokeExact(
                modelH,
                textSeg,
                textLen,
                outTokensSeg,
                (int) maxTokens,
                addBos
            );
            if (res == Integer.MIN_VALUE) {
                throw new ArgusNativeException(ArgusNativeException.ERROR_INTERNAL,
                    "Tokenization overflow: result size exceeds int32 limit");
            }
            if (res < 0) {
                int lastErr = (int) ArgusBindings.argus_last_error_code.invokeExact();
                if (lastErr != 0) {
                    ArgusNativeException.checkStatus(res, "argus_tokenize_n");
                }
                // When lastErr == 0, res is -required_tokens indicating insufficient buffer capacity
            }
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to tokenize input text", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Converts a Java string into vocabulary token IDs using this context's retained model.
     * Automatically handles buffer sizing and allocation within the provided arena.
     *
     * @param arena Arena used to allocate the output token segment
     * @param text Input text string to tokenize
     * @param addBos Whether to prepend the beginning-of-sequence (BOS) token
     * @return MemorySegment containing the token integers (int32)
     */
    public MemorySegment tokenize(Arena arena, String text, boolean addBos) {
        Objects.requireNonNull(arena, "arena");
        Objects.requireNonNull(text, "text");
        MemorySegment textSeg = ArgusValidation.allocateUtf8Text(arena, text);
        int initialCap = Math.max(64, (int) textSeg.byteSize());
        MemorySegment tokenBuf = arena.allocate(ValueLayout.JAVA_INT, initialCap);
        int nTokens = tokenize(textSeg, tokenBuf, addBos);
        if (nTokens < 0) {
            if (nTokens == Integer.MIN_VALUE) {
                throw new ArgusNativeException(ArgusNativeException.ERROR_INTERNAL,
                    "Tokenization overflow: result size exceeds int32 limit");
            }
            int required = -nTokens;
            if (required <= 0 || required > 10_000_000) {
                throw new IllegalArgumentException("Token count exceeds safety limit: " + required);
            }
            tokenBuf = arena.allocate(ValueLayout.JAVA_INT, required);
            nTokens = tokenize(textSeg, tokenBuf, addBos);
        }
        if (nTokens < 0) {
            ArgusNativeException.checkStatus(nTokens, "argus_tokenize_n");
        }
        return tokenBuf.asSlice(0, (long) nTokens * ValueLayout.JAVA_INT.byteSize());
    }

    /**
     * Converts a Java string into vocabulary token IDs as a primitive Java int array.
     *
     * @param text Input text string to tokenize
     * @param addBos Whether to prepend the beginning-of-sequence (BOS) token
     * @return Primitive int array containing the token IDs
     */
    public int[] tokenize(String text, boolean addBos) {
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment tokenSeg = tokenize(localArena, text, addBos);
            int count = (int) (tokenSeg.byteSize() / ValueLayout.JAVA_INT.byteSize());
            int[] result = new int[count];
            MemorySegment.copy(tokenSeg, ValueLayout.JAVA_INT, 0, result, 0, count);
            return result;
        }
    }

    /**
     * Converts a single token ID into its raw UTF-8 byte sequence using the retained model reference.
     * Supports arbitrary-length token pieces via 2-tier buffer resizing.
     */
    public byte[] tokenToPieceBytes(int token) {
        MemorySegment ctxH = acquireReadLease();
        try (Arena local = Arena.ofConfined()) {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int bufSize = 256;
            MemorySegment bufSeg = local.allocate(bufSize);

            int result = (int) ArgusBindings.argus_token_to_piece.invokeExact(
                modelH,
                token,
                bufSeg,
                bufSize
            );
            if (result < 0 && result != Integer.MIN_VALUE) {
                int lastErr = (int) ArgusBindings.argus_last_error_code.invokeExact();
                if (lastErr == 0) {
                    int required = -result;
                    if (required > 0 && required <= 16 * 1024 * 1024) {
                        bufSeg = local.allocate(required);
                        result = (int) ArgusBindings.argus_token_to_piece.invokeExact(
                            modelH,
                            token,
                            bufSeg,
                            required
                        );
                    }
                }
            }
            if (result <= 0) {
                return new byte[0];
            }
            byte[] bytes = new byte[result];
            MemorySegment.copy(bufSeg, ValueLayout.JAVA_BYTE, 0,
                MemorySegment.ofArray(bytes), ValueLayout.JAVA_BYTE, 0, result);
            return bytes;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to convert token " + token + " to piece bytes", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Converts a single token ID to its text representation (piece).
     * Operates on the execution context's retained model reference, allowing this method
     * to continue functioning even if the original Java ArgusModel wrapper has been closed.
     * This operation is lock-free and read-only.
     */
    public String tokenToPiece(int token) {
        byte[] bytes = tokenToPieceBytes(token);
        if (bytes.length == 0) {
            return "";
        }
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * Converts a single token ID into raw UTF-8 bytes written directly into a caller-supplied byte array.
     */
    public int tokenToPieceInto(int token, byte[] dst, int offset, int maxLen) {
        Objects.requireNonNull(dst, "dst");
        if (offset < 0 || maxLen <= 0 || (long) offset + (long) maxLen > dst.length) {
            throw new IllegalArgumentException("Invalid slice bounds: offset=" + offset + ", maxLen=" + maxLen + ", dst.length=" + dst.length);
        }
        MemorySegment ctxH = acquireReadLease();
        try (Arena local = Arena.ofConfined()) {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            MemorySegment bufSeg = local.allocate(maxLen);
            int res = (int) ArgusBindings.argus_token_to_piece.invokeExact(modelH, token, bufSeg, maxLen);
            if (res > 0) {
                MemorySegment.copy(bufSeg, ValueLayout.JAVA_BYTE, 0, dst, offset, res);
            }
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to convert token " + token + " into byte array", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Synchronously decodes a batch of tokens in the context.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     *
     * <p><b>KV-Cache Window Contract:</b> {@code startPos + nTokens} must not exceed the
     * configured context window ({@code nCtx}), and the sequence slot must have sufficient
     * free KV-cache capacity. For long-running or streaming generation approaching {@code nCtx},
     * callers can implement a sliding-window ring buffer without resetting sequence state:
     * <ol>
     *   <li>Evict an older span of tokens after any pinned prefix via {@link #clearCacheSlot(int, int, int)}.</li>
     *   <li>If {@link #canShift()} is {@code true}, shift the surviving suffix positions left via
     *       {@link #shiftCacheSlot(int, int, int, int)} to reclaim contiguous position space.</li>
     * </ol>
     *
     * @param tokensSeg     tokens to evaluate
     * @param nTokens       number of tokens
     * @param startPos      KV cache offset position
     * @param seqId         sequence tracking ID
     * @param requestLogits true to evaluate logits on the final token of the batch
     * @return 0 on success, non-zero on failure
     */
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits) {
        return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, MemorySegment.NULL);
    }

    /**
     * Synchronously decodes a batch of tokens in the context with optional early cancellation.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     * See {@link #decodeBatch(MemorySegment, int, int, int, boolean)} for the KV-cache window contract.
     *
     * @param tokensSeg     tokens to evaluate
     * @param nTokens       number of tokens
     * @param startPos      KV cache offset position
     * @param seqId         sequence tracking ID
     * @param requestLogits true to evaluate logits on the final token of the batch
     * @param abortFlag     optional safe cancellation flag object
     * @return 0 on success, -2 if aborted, non-zero on failure
     */
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        if (abortFlag != null) {
            try (var lease = abortFlag.lease()) {
                return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, lease.handle());
            }
        }
        return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, MemorySegment.NULL);
    }

    /**
     * Synchronously decodes a batch of tokens in the context with optional early cancellation.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     *
     * @param tokensSeg     tokens to evaluate
     * @param nTokens       number of tokens
     * @param startPos      KV cache offset position
     * @param seqId         sequence tracking ID
     * @param requestLogits true to evaluate logits on the final token of the batch
     * @param abortFlagSeg  raw memory segment targeting argus_abort_flag_t; caller is responsible
     *                      for ensuring the underlying native struct remains valid for the downcall duration
     * @return 0 on success, -2 if aborted, non-zero on failure
     * @deprecated Use {@link #decodeBatch(MemorySegment, int, int, int, boolean, ArgusAbortFlag)}
     *             which enforces lease-managed lifetime safety.
     */
    @Deprecated(since = "1.9.1")
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits, MemorySegment abortFlagSeg) {
        Objects.requireNonNull(tokensSeg);
        Objects.requireNonNull(abortFlagSeg);
        if (nTokens == 0) {
            return 0;
        }

        long requiredBytes = ArgusValidation.multiplyExactBytes(nTokens, ValueLayout.JAVA_INT.byteSize(), "tokensSeg");
        ArgusValidation.checkReadable(tokensSeg, requiredBytes, "tokensSeg");
        ArgusValidation.checkAlignment(tokensSeg, Integer.BYTES, "tokensSeg");
        ArgusValidation.checkNonNegative(startPos, "startPos");
        ArgusValidation.checkNonNegative(seqId, "seqId");

        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_decode_tokens.invokeExact(
                ctxH, tokensSeg, nTokens, startPos, seqId, requestLogits, abortFlagSeg);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to decode batch of tokens", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Evaluates tokenized multimodal chunks (combining text and media projectors).
     *
     * @param mctx        loaded multimodal context
     * @param chunks      tokenized prompt chunks
     * @param nPast       current KV cache offset position
     * @param seqId       target sequence tracking ID
     * @param nBatch      batch processing size constraint
     * @param logitsLast  true to request logits only on the final token
     * @return the updated KV cache offset position (new n_past)
     */
    public int evalMultimodalChunks(ArgusMultimodalContext mctx, ArgusInputChunks chunks, int nPast, int seqId, int nBatch, boolean logitsLast) {
        Objects.requireNonNull(mctx);
        Objects.requireNonNull(chunks);
        ArgusValidation.checkNonNegative(nPast, "nPast");
        ArgusValidation.checkNonNegative(seqId, "seqId");
        ArgusValidation.checkPositive(nBatch, "nBatch");

        MemorySegment ctxH = acquireReadLease();
        try {
            try (var mctxLease = mctx.lease();
                 var chunksLease = chunks.lease();
                 Arena local = Arena.ofConfined()) {
                MemorySegment outNewNPast = local.allocate(ValueLayout.JAVA_INT);
                int res = (int) ArgusBindings.argus_eval_multimodal_chunks.invokeExact(
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

                return outNewNPast.get(ValueLayout.JAVA_INT, 0);
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to evaluate multimodal chunks", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Checks if speculative decoding draft context is initialized and active on this context session.
     *
     * @return true if speculative draft context is active, false otherwise
     */
    public boolean hasDraftContext() {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_context_has_draft.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query draft context status", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Samples the next token ID from the last computed logits matrix using default parameters.
     * Mutex-locked inside the native layer with zero-allocation caching.
     */
    public int sampleToken(int seqId, float temperature, float repeatPenalty) {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_sample_token.invokeExact(ctxH, seqId, temperature, repeatPenalty);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to sample token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Samples the next token ID using the extended sampling configuration.
     * Mutex-locked inside the native layer with zero-allocation caching.
     *
     * @param seqId  sequence ID
     * @param config extended sampling configuration parameters (temperature, penalties, top_p, min_p, top_k, dry)
     * @return the sampled token ID
     */
    public int sampleToken(int seqId, ArgusSamplerConfig config) {
        return sampleTokenWithBias(seqId, config, MemorySegment.NULL, 0);
    }

    /**
     * Samples the next token ID from the last computed logits matrix, applying logit steering biases.
     * Enforces strict zero-allocation boundaries via a pre-allocated unmanaged struct segment.
     *
     * @param seqId              sequence ID
     * @param temperature        sampling temperature
     * @param repeatPenalty      repetition penalty
     * @param biasSegment        off-heap contiguous segment of argus_logit_bias_t structs (see ArgusLayouts.LOGIT_BIAS)
     * @param biasCount          total count of biased tokens in the segment
     * @return the sampled token ID
     */
    public int sampleTokenWithBias(int seqId, float temperature, float repeatPenalty,
                                   MemorySegment biasSegment, int biasCount) {
        Objects.requireNonNull(biasSegment);
        ArgusValidation.checkNonNegative(biasCount, "biasCount");
        if (biasCount > 0) {
            long requiredBytes = ArgusValidation.multiplyExactBytes(biasCount, ArgusLayouts.LOGIT_BIAS.byteSize(), "biasSegment");
            ArgusValidation.checkReadable(biasSegment, requiredBytes, "biasSegment");
        }
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_sample_token_with_bias.invokeExact(
                ctxH, seqId, temperature, repeatPenalty, biasSegment, biasCount
            );
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to sample token with bias", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Samples the next token ID using the extended sampling configuration and logit biases.
     * Mutex-locked inside the native layer with zero-allocation caching.
     *
     * @param seqId       sequence ID
     * @param config      extended sampling configuration parameters
     * @param biasSegment off-heap contiguous segment of argus_logit_bias_t structs (or MemorySegment.NULL)
     * @param biasCount   total count of biased tokens in the segment
     * @return the sampled token ID
     */
    public int sampleTokenWithBias(int seqId, ArgusSamplerConfig config, MemorySegment biasSegment, int biasCount) {
        Objects.requireNonNull(config);
        ArgusValidation.checkNonNegative(biasCount, "biasCount");
        MemorySegment biasSeg = (biasSegment != null) ? biasSegment : MemorySegment.NULL;
        if (biasCount > 0) {
            long requiredBytes = ArgusValidation.multiplyExactBytes(biasCount, ArgusLayouts.LOGIT_BIAS.byteSize(), "biasSegment");
            ArgusValidation.checkReadable(biasSeg, requiredBytes, "biasSegment");
        }

        MemorySegment ctxH = acquireReadLease();
        try {
            samplerLock.lock();
            try {
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("temperature")),
                    config.temperature()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("repeat_penalty")),
                    config.repeatPenalty()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_INT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("repeat_last_n")),
                    config.repeatLastN()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("frequency_penalty")),
                    config.frequencyPenalty()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("presence_penalty")),
                    config.presencePenalty()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("top_p")),
                    config.topP()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("min_p")),
                    config.minP()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_INT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("top_k")),
                    config.topK()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("dry_multiplier")),
                    config.dryMultiplier()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_FLOAT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("dry_base")),
                    config.dryBase()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_INT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("dry_allowed_length")),
                    config.dryAllowedLength()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_INT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("dry_penalty_last_n")),
                    config.dryPenaltyLastN()
                );
                samplerParamsSeg.set(ValueLayout.JAVA_INT,
                    ArgusLayouts.SAMPLER_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("seed")),
                    config.seed()
                );

                return (int) ArgusBindings.argus_sample_token_ext.invokeExact(
                    ctxH, seqId, samplerParamsSeg, biasSeg, biasCount
                );
            } finally {
                samplerLock.unlock();
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to sample token with extended config", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Resets the sampler state and clears penalty/DRY token history for a sequence slot.
     *
     * @param seqId sequence ID (-1 for all sequences)
     */
    public void resetSampler(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_reset.invokeExact(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_reset");
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to reset sampler for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Primes a sequence slot's sampler history with prompt or external tokens.
     * Accepted tokens will be tracked for repetition penalties and DRY n-gram suppression.
     *
     * @param seqId     sequence ID
     * @param tokensSeg off-heap segment containing 4-byte int tokens
     * @param nTokens   number of tokens
     * @return 0 on success, negative on failure
     */
    public int primeSampler(int seqId, MemorySegment tokensSeg, int nTokens) {
        Objects.requireNonNull(tokensSeg);
        ArgusValidation.checkNonNegative(nTokens, "nTokens");
        if (nTokens == 0) {
            return 0;
        }
        long requiredBytes = ArgusValidation.multiplyExactBytes(nTokens, ValueLayout.JAVA_INT.byteSize(), "tokensSeg");
        ArgusValidation.checkReadable(tokensSeg, requiredBytes, "tokensSeg");

        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_prime.invokeExact(ctxH, seqId, tokensSeg, nTokens);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_prime");
            }
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to prime sampler for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Truncates a sequence slot's sampler token history to the specified length.
     * Resets the sampler chain and replays the surviving prefix.
     *
     * @param seqId     sequence ID
     * @param newLength retained token history length
     */
    public void truncateSampler(int seqId, int newLength) {
        ArgusValidation.checkNonNegative(newLength, "newLength");
        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_truncate.invokeExact(ctxH, seqId, newLength);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_truncate");
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to truncate sampler for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries the number of history tokens (primed + committed) retained in a sequence slot's sampler.
     *
     * @param seqId sequence ID
     * @return count of retained tokens
     */
    public int getSamplerHistoryCount(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_get_history_count.invokeExact(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_get_history_count");
            }
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query sampler history count for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries whether a sequence slot has an uncommitted pending sample awaiting decode.
     *
     * @param seqId sequence ID
     * @return true if a pending sample exists, false otherwise
     */
    public boolean hasSamplerPending(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_has_pending.invokeExact(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_has_pending");
            }
            return res == 1;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query sampler pending status for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Explicitly discards an uncommitted pending sample from a sequence slot.
     * Purges the pending token from internal sampler penalty filters via chain rebuild,
     * preserves the active RNG stream, and leaves committed history and KV cache untouched.
     *
     * @param seqId sequence ID
     * @return true if a pending sample was discarded and filters rebuilt, false if no pending sample existed
     */
    public boolean discardPendingSample(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_sampler_discard_pending.invokeExact(ctxH, seqId);
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_sampler_discard_pending");
            }
            return res == 1;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to discard pending sample for sequence " + seqId, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Removes/prunes a segment from the KV cache sequence tracking lists.
     * Mutex-locked inside the native layer.
     *
     * @param seqId sequence ID (negative matches any sequence)
     * @param p0    start position (negative is [0, p1])
     * @param p1    end position (negative is [p0, inf))
     * @return true if successfully cleared/pruned, false if rejected or invalid
     */
    public boolean clearCacheSlot(int seqId, int p0, int p1) {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_kv_cache_clear_slot.invokeExact(ctxH, seqId, p0, p1);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to prune KV cache sequence slot", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Copies all tokens and recurrent hidden states from a source sequence slot to a destination sequence slot,
     * replicating the persistent sampler history and RNG distribution state machine.
     *
     * @param srcSeqId source sequence tracking ID
     * @param dstSeqId destination sequence tracking ID
     * @param p0       start position offset (negative is 0)
     * @param p1       end position offset (negative is infinity)
     * @return true on success, false on invalid parameters or architectural rejection
     */
    public boolean copySequenceSlot(int srcSeqId, int dstSeqId, int p0, int p1) {
        ArgusValidation.checkNonNegative(srcSeqId, "srcSeqId");
        ArgusValidation.checkNonNegative(dstSeqId, "dstSeqId");
        MemorySegment ctxH = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_kv_cache_seq_cp.invokeExact(ctxH, srcSeqId, dstSeqId, p0, p1);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to copy KV cache sequence slot", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Copies entire prefix sequence from source sequence slot to destination sequence slot [0, -1).
     *
     * @param srcSeqId source sequence tracking ID
     * @param dstSeqId destination sequence tracking ID
     * @return true on success, false on invalid parameters or architectural rejection
     */
    public boolean copySequenceSlot(int srcSeqId, int dstSeqId) {
        return copySequenceSlot(srcSeqId, dstSeqId, 0, -1);
    }

    /**
     * Checks whether the active KV cache memory supports position shifting (sliding window attention).
     */
    public boolean canShift() {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_kv_cache_can_shift.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query KV cache can_shift", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Adds an integer position delta to all KV-cache tokens in {@code [p0, p1)} for a sequence slot.
     * Used after {@link #clearCacheSlot(int, int, int)} to shift remaining KV-cache positions left
     * (with a negative {@code delta}) and reclaim contiguous position space within {@code nCtx}.
     * Also synchronizes speculative draft KV-cache positions and coordinate-linked sampler history.
     *
     * @param seqId sequence tracking ID (must be in {@code [0, getSeqMax())})
     * @param p0    start position inclusive (negative is {@code 0})
     * @param p1    end position exclusive (negative is infinity)
     * @param delta relative position shift to add (e.g. {@code -evictedCount})
     * @return true if positions were shifted (or {@code delta == 0} no-op), false on invalid bounds or unsupported architecture
     */
    public boolean shiftCacheSlot(int seqId, int p0, int p1, int delta) {
        ArgusValidation.checkNonNegative(seqId, "seqId");
        MemorySegment ctxH = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_kv_cache_seq_add.invokeExact(ctxH, seqId, p0, p1, delta);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to shift KV cache sequence slot positions", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries the maximum number of sequence slots allocated for this execution context.
     */
    public int getSeqMax() {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_context_get_seq_max.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query context seq_max", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Atomically leases an available sequence slot from the configured clone pool,
     * copies the state from {@code srcSeqId} into the leased slot (including KV cache,
     * persistent sampler history, pending sample state, and distribution RNG continuity),
     * and returns the newly leased slot index.
     *
     * @param srcSeqId source sequence slot index to clone from (e.g. 0 for prompt template)
     * @return leased sequence slot index
     * @throws IllegalStateException if the clone pool is unconfigured or all clone slots are leased
     */
    public int forkSlot(int srcSeqId) {
        return slotPool.forkSlot(srcSeqId, getSeqMax(), this::copySequenceSlot);
    }

    /**
     * Clears a previously leased clone sequence slot and returns it to the free slot pool.
     *
     * @param seqId sequence slot index to release (must belong to the configured clone pool)
     */
    public void freeSlot(int seqId) {
        slotPool.freeSlot(seqId, slot -> clearCacheSlot(slot, 0, -1));
    }

    /**
     * Returns the number of unleased sequence slots currently available for forking.
     */
    public int getAvailableSlotCount() {
        return slotPool.getAvailableSlotCount();
    }

    /**
     * Checks whether the underlying model uses a hybrid attention-recurrent architecture.
     */
    public boolean isHybrid() {
        return modelRef.isHybrid();
    }

    /**
     * Checks whether the underlying model uses a recurrent architecture.
     */
    public boolean isRecurrent() {
        return modelRef.isRecurrent();
    }

    /**
     * Checks whether the underlying model uses a diffusion architecture.
     */
    public boolean isDiffusion() {
        return modelRef.isDiffusion();
    }

    /**
     * Queries the highest position index currently allocated in the KV cache for a sequence slot.
     *
     * @param seqId sequence tracking ID
     * @return largest position index in memory, or -1 if sequence slot is empty or invalid
     */
    public int getSeqPosMax(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_kv_cache_seq_pos_max.invokeExact(ctxH, seqId);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query KV cache seq_pos_max", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries the lowest position index currently allocated in the KV cache for a sequence slot.
     *
     * @param seqId sequence tracking ID
     * @return smallest position index in memory, or -1 if sequence slot is empty or invalid
     */
    public int getSeqPosMin(int seqId) {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_kv_cache_seq_pos_min.invokeExact(ctxH, seqId);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query KV cache seq_pos_min", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Synthesizes audio samples using WavTokenizer and OuteTTS pipelines.
     * Mutex-locked inside the native layer.
     *
     * @param wavtokenizerModel GGUF weights of the WavTokenizer vocoder model
     * @param textSeg           input text prompt MemorySegment (UTF-8 encoded)
     * @param voiceSeed         seed controlling speaker characteristics
     * @param outPcmSeg         destination buffer to write 24kHz float PCM audio into
     * @param maxSamples        capacity of output PCM buffer in floats
     * @return sample count generated, or negative on failure
     */
    public int synthesizeSpeech(ArgusModel wavtokenizerModel, MemorySegment textSeg, int voiceSeed, MemorySegment outPcmSeg, int maxSamples) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outPcmSeg);
        ArgusValidation.checkNonNegative(maxSamples, "maxSamples");
        if (maxSamples == 0) {
            return 0;
        }
        ArgusValidation.checkReadable(textSeg, 0, "textSeg");
        long requiredPcmBytes = ArgusValidation.multiplyExactBytes(maxSamples, ValueLayout.JAVA_FLOAT.byteSize(), "outPcmSeg");
        ArgusValidation.checkWritable(outPcmSeg, requiredPcmBytes, "outPcmSeg");

        MemorySegment ctxH = acquireReadLease();
        try {
            ArgusNativeResource.Lease wavLease = (wavtokenizerModel != null) ? wavtokenizerModel.lease() : null;
            try {
                MemorySegment wavModelHandle = (wavLease != null) ? wavLease.handle() : MemorySegment.NULL;
                ttsLock.lock();
                try {
                    long textLen = textSeg.byteSize();
                    int res = (int) ArgusBindings.argus_synthesize_speech_n.invokeExact(
                        ctxH,
                        wavModelHandle,
                        textSeg,
                        textLen,
                        voiceSeed,
                        outPcmSeg,
                        maxSamples,
                        ttsWorkspace,
                        ttsWorkspaceSizeFloats
                    );

                    if (res < -1) {
                        // Resize off-heap workspace buffer with 64-byte SIMD cache line alignment
                        long requiredFloats = -res;
                        this.ttsWorkspace = contextArena.allocate(
                            requiredFloats * ValueLayout.JAVA_FLOAT.byteSize(),
                            64
                        );
                        this.ttsWorkspaceSizeFloats = requiredFloats;

                        // Retry execution
                        res = (int) ArgusBindings.argus_synthesize_speech_n.invokeExact(
                            ctxH,
                            wavModelHandle,
                            textSeg,
                            textLen,
                            voiceSeed,
                            outPcmSeg,
                            maxSamples,
                            ttsWorkspace,
                            ttsWorkspaceSizeFloats
                        );
                    }

                    if (res < 0) {
                        ArgusNativeException.checkStatus(res, "argus_synthesize_speech_n");
                    }

                    return res;
                } finally {
                    ttsLock.unlock();
                }
            } finally {
                if (wavLease != null) {
                    wavLease.close();
                }
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to synthesize speech", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Retrieves the embedding vector for a specific sequence ID.
     * Mutex-locked inside the native layer.
     *
     * @param seqId          target sequence tracking ID
     * @param outEmbeddings  destination buffer memory segment to receive the float embedding vector
     * @param maxFloats      maximum capacity of the output buffer in float elements
     * @return the number of float elements written, or negative on failure
     */
    public int getEmbeddings(int seqId, MemorySegment outEmbeddings, int maxFloats) {
        Objects.requireNonNull(outEmbeddings);
        ArgusValidation.checkNonNegative(maxFloats, "maxFloats");
        if (maxFloats == 0) {
            return 0;
        }
        long requiredBytes = ArgusValidation.multiplyExactBytes(maxFloats, ValueLayout.JAVA_FLOAT.byteSize(), "outEmbeddings");
        ArgusValidation.checkWritable(outEmbeddings, requiredBytes, "outEmbeddings");

        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_get_embeddings.invokeExact(ctxH, seqId, outEmbeddings, maxFloats);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve embeddings from native context", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Retrieves the pooled or token embeddings evaluated for default sequence 0 into the specified float buffer.
     * Mutex-locked inside the native layer.
     *
     * @param outEmbeddings destination buffer memory segment to receive the float embedding vector
     * @param maxFloats maximum capacity of the output buffer in float elements
     * @return the number of float elements written
     * @throws ArgusNativeException if retrieving embeddings fails
     */
    public int getEmbeddings(MemorySegment outEmbeddings, int maxFloats) {
        return getEmbeddings(0, outEmbeddings, maxFloats);
    }


    /**
     * Dynamically configures the CPU thread counts for this execution context session.
     * Automatically updates main evaluation, speculative draft, and vocoder contexts if present.
     *
     * @param nThreads      number of threads for single-token generation decoding
     * @param nThreadsBatch number of threads for prompt and batch token processing
     */
    public void setNThreads(int nThreads, int nThreadsBatch) {
        ArgusValidation.checkPositive(nThreads, "nThreads");
        ArgusValidation.checkPositive(nThreadsBatch, "nThreadsBatch");
        MemorySegment ctxH = acquireReadLease();
        try {
            ArgusBindings.argus_set_n_threads.invokeExact(ctxH, nThreads, nThreadsBatch);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to set thread counts for context", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries active CPU thread count for single-token generation decoding.
     *
     * @return number of allocated generation threads
     */
    public int getNThreads() {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_get_n_threads.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query thread count for context", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Queries active CPU thread count for prompt and batch token processing.
     *
     * @return number of allocated batch threads
     */
    public int getNThreadsBatch() {
        MemorySegment ctxH = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_get_n_threads_batch.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query batch thread count for context", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Retrieves unmanaged performance execution telemetry for this context session.
     *
     * @return populated {@link ArgusPerfTimings} instance
     */
    public ArgusPerfTimings getPerfTimings() {
        MemorySegment ctxH = acquireReadLease();
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment outSeg = localArena.allocate(cc.projectargus.libargus.internal.ArgusLayouts.PERF_TIMINGS);
            boolean ok = (boolean) ArgusBindings.argus_context_get_perf.invokeExact(ctxH, outSeg);
            if (!ok) {
                ArgusNativeException.throwLastError("argus_context_get_perf");
            }
            double startMs = outSeg.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 0);
            double loadMs = outSeg.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 8);
            double pEvalMs = outSeg.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 16);
            double evalMs = outSeg.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 24);
            int nPromptEval = outSeg.get(java.lang.foreign.ValueLayout.JAVA_INT, 32);
            int nEval = outSeg.get(java.lang.foreign.ValueLayout.JAVA_INT, 36);
            int nReused = outSeg.get(java.lang.foreign.ValueLayout.JAVA_INT, 40);
            int nReallocsSame = outSeg.get(java.lang.foreign.ValueLayout.JAVA_INT, 44);
            int nReallocsDiff = outSeg.get(java.lang.foreign.ValueLayout.JAVA_INT, 48);

            return new ArgusPerfTimings(startMs, loadMs, pEvalMs, evalMs, nPromptEval, nEval, nReused, nReallocsSame, nReallocsDiff);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query context performance timings", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Resets performance execution telemetry counters on this context session.
     */
    public void resetPerfTimings() {
        MemorySegment ctxH = acquireReadLease();
        try {
            ArgusBindings.argus_context_reset_perf.invokeExact(ctxH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to reset context performance timings", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Configures whether this context uses causal attention masking.
     *
     * @param causal True for causal masking (autoregressive models), false for bidirectional attention (embedding models).
     */
    public void setCausalAttn(boolean causal) {
        MemorySegment ctxH = acquireReadLease();
        try {
            boolean ok = (boolean) ArgusBindings.argus_context_set_causal_attn.invokeExact(ctxH, causal);
            if (!ok) {
                ArgusNativeException.throwLastError("argus_context_set_causal_attn");
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to set causal attention", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns true if this context is currently configured for causal attention masking.
     */
    public boolean isCausalAttn() {
        MemorySegment ctxH = acquireReadLease();
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_BOOLEAN);
            boolean ok = (boolean) ArgusBindings.argus_context_get_causal_attn.invokeExact(ctxH, outSeg);
            if (!ok) {
                ArgusNativeException.throwLastError("argus_context_get_causal_attn");
            }
            return outSeg.get(ValueLayout.JAVA_BOOLEAN, 0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query causal attention status", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the effective output embedding dimension size for the model retained by this execution context.
     * Operates safely even if the original Java ArgusModel wrapper has been closed.
     *
     * @return Model output embedding dimension in floats (e.g. 32, 768, 1024, 4096)
     */
    public int nEmbd() {
        MemorySegment ctxH = acquireReadLease();
        try {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = (int) ArgusBindings.argus_model_n_embd.invokeExact(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd");
            }
            return dim;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model embedding dimension", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the explicit output embedding dimension size for the model retained by this execution context.
     * Equivalent to {@link #nEmbd()}.
     *
     * @return Model output embedding dimension in floats
     */
    public int nEmbdOut() {
        MemorySegment ctxH = acquireReadLease();
        try {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = (int) ArgusBindings.argus_model_n_embd_out.invokeExact(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd_out");
            }
            return dim;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model output embedding dimension", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the internal input/backbone embedding dimension size ({@code n_embd_inp}) for the model
     * retained by this execution context.
     *
     * @return Model internal input embedding dimension in floats
     */
    public int nEmbdInp() {
        MemorySegment ctxH = acquireReadLease();
        try {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            int dim = (int) ArgusBindings.argus_model_n_embd_inp.invokeExact(modelH);
            if (dim <= 0) {
                ArgusNativeException.throwLastError("argus_model_n_embd_inp");
            }
            return dim;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model input embedding dimension", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Pure zero-allocation steady-state embedding evaluation pipeline.
     * Evaluates a pre-tokenized token segment on sequence slot 0 and retrieves the embedding vector
     * directly into the caller's preallocated native memory segment under a single atomic context lock.
     *
     * @param tokensSeg Native MemorySegment containing token integers (must be 4-byte aligned)
     * @param nTokens Number of tokens in tokensSeg to evaluate
     * @param outEmbeddings Destination MemorySegment allocated for at least {@link #nEmbd()} floats
     * @return Number of floats written to outEmbeddings
     */
    public int evalEmbeddings(MemorySegment tokensSeg, int nTokens, MemorySegment outEmbeddings) {
        return evalEmbeddings(tokensSeg, nTokens, 0, outEmbeddings);
    }

    /**
     * Evaluates a pre-tokenized token segment on a specified sequence slot and retrieves the embedding vector
     * directly into the caller's preallocated native memory segment under a single atomic context lock.
     *
     * @param tokensSeg Native MemorySegment containing token integers (must be 4-byte aligned)
     * @param nTokens Number of tokens in tokensSeg to evaluate
     * @param seqId Target sequence ID
     * @param outEmbeddings Destination MemorySegment allocated for at least {@link #nEmbd()} floats
     * @return Number of floats written to outEmbeddings
     */
    public int evalEmbeddings(MemorySegment tokensSeg, int nTokens, int seqId, MemorySegment outEmbeddings) {
        Objects.requireNonNull(tokensSeg, "tokensSeg");
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        ArgusValidation.checkReadable(tokensSeg, (long) nTokens * ValueLayout.JAVA_INT.byteSize(), "tokensSeg");
        ArgusValidation.checkWritable(outEmbeddings, ValueLayout.JAVA_FLOAT.byteSize(), "outEmbeddings");
        if (nTokens <= 0) {
            throw new IllegalArgumentException("nTokens must be positive, got: " + nTokens);
        }

        int requiredDim = nEmbd();
        long maxFloats = outEmbeddings.byteSize() / ValueLayout.JAVA_FLOAT.byteSize();
        if (maxFloats < requiredDim) {
            throw new IllegalArgumentException("outEmbeddings capacity (" + maxFloats +
                " floats) is insufficient for model embedding dimension (" + requiredDim + " floats)");
        }

        MemorySegment ctxH = acquireReadLease();
        try {
            int written = (int) ArgusBindings.argus_eval_embeddings.invokeExact(
                ctxH,
                tokensSeg,
                nTokens,
                seqId,
                outEmbeddings,
                (int) maxFloats
            );
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_eval_embeddings");
            }
            if (written != requiredDim) {
                throw new IllegalStateException("Incomplete embedding extraction: expected " + requiredDim + " floats, got " + written);
            }
            return written;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to evaluate embeddings atomically", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Convenience method to extract semantic text embeddings into the caller's preallocated MemorySegment,
     * utilizing the caller-supplied Arena for intermediate token scratch buffers.
     * Tokenizes prompt text, decodes a single batch, and retrieves the embedding vector atomically.
     *
     * @param arena Confined or temporary arena used for intermediate token buffers
     * @param text Input text string to embed
     * @param outEmbeddings Destination MemorySegment allocated for at least {@link #nEmbd()} floats
     * @return Number of floats written to outEmbeddings
     */
    public int extractEmbeddings(Arena arena, String text, MemorySegment outEmbeddings) {
        Objects.requireNonNull(arena, "arena");
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        if (text.isEmpty()) {
            throw new IllegalArgumentException("Cannot extract embeddings for empty text string");
        }

        MemorySegment textSeg = ArgusValidation.allocateUtf8Text(arena, text);
        int maxTokens = Math.max(512, (int) textSeg.byteSize());
        MemorySegment tokenBuf = arena.allocate(ValueLayout.JAVA_INT, maxTokens);
        int nTokens = tokenize(textSeg, tokenBuf, true);
        if (nTokens < 0) {
            if (nTokens == Integer.MIN_VALUE) {
                throw new ArgusNativeException(ArgusNativeException.ERROR_INTERNAL,
                    "Tokenization overflow: result size exceeds int32 limit");
            }
            int requiredTokens = -nTokens;
            if (requiredTokens <= 0 || requiredTokens > 10_000_000) {
                throw new IllegalArgumentException("Token count exceeds safety limit: " + requiredTokens);
            }
            tokenBuf = arena.allocate(ValueLayout.JAVA_INT, requiredTokens);
            nTokens = tokenize(textSeg, tokenBuf, true);
        }
        if (nTokens <= 0) {
            throw new IllegalStateException("Tokenization produced 0 tokens for non-empty text");
        }

        return evalEmbeddings(tokenBuf, nTokens, outEmbeddings);
    }

    /**
     * Returns the underlying primary model reference associated with this context.
     */
    public ArgusModel getModel() {
        return modelRef;
    }

    /**
     * Convenience method to extract semantic text embeddings as a standard Java float array.
     *
     * @param text Input text to embed
     * @return Float array containing the embedding vector
     */
    public float[] extractEmbeddings(String text) {
        Objects.requireNonNull(text, "text");
        int dim = nEmbd();
        float[] result = new float[dim];
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_FLOAT, dim);
            int written = extractEmbeddings(localArena, text, outSeg);
            if (written != dim) {
                throw new IllegalStateException("Incomplete embedding extraction: expected " + dim + " floats, got " + written);
            }
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, result, 0, written);
        }
        return result;
    }

    /**
     * Checks if the given token ID is an End-Of-Generation (EOG) token using the context's retained model.
     */
    public boolean vocabIsEog(int token) {
        MemorySegment ctxH = acquireReadLease();
        try {
            MemorySegment modelH = (MemorySegment) ArgusBindings.argus_context_get_model.invokeExact(ctxH);
            if (modelH.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Execution context has no retained model");
            }
            return (boolean) ArgusBindings.argus_vocab_is_eog.invokeExact(modelH, token);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check EOG token: " + token, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Synchronously decodes a batch of tokens from a primitive Java int array.
     */
    public int decodeBatch(int[] tokens, int nTokens, int startPos, int seqId, boolean requestLogits) {
        return decodeBatch(tokens, 0, nTokens, startPos, seqId, requestLogits, null);
    }

    /**
     * Synchronously decodes a batch of tokens from a primitive Java int array with optional early cancellation.
     */
    public int decodeBatch(int[] tokens, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        return decodeBatch(tokens, 0, nTokens, startPos, seqId, requestLogits, abortFlag);
    }

    /**
     * Synchronously decodes a slice of tokens from a primitive Java int array with optional early cancellation.
     */
    public int decodeBatch(int[] tokens, int offset, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(tokens, "tokens");
        ArgusValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
        if (nTokens == 0) {
            return 0;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment tokensSeg = localArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment.copy(tokens, offset, tokensSeg, ValueLayout.JAVA_INT, 0, nTokens);
            return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, abortFlag);
        }
    }

    /**
     * Samples the next token ID applying logit biases from primitive arrays.
     */
    public int sampleTokenWithBias(int seqId, float temperature, float repeatPenalty,
                                   int[] biasTokens, float[] biasValues, int biasCount) {
        ArgusValidation.checkNonNegative(biasCount, "biasCount");
        if (biasCount == 0) {
            return sampleToken(seqId, temperature, repeatPenalty);
        }
        Objects.requireNonNull(biasTokens, "biasTokens");
        Objects.requireNonNull(biasValues, "biasValues");
        if (biasCount > biasTokens.length || biasCount > biasValues.length) {
            throw new IllegalArgumentException("biasCount exceeds bias array length");
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment biasSeg = localArena.allocate(ArgusLayouts.LOGIT_BIAS, biasCount);
            long stride = ArgusLayouts.LOGIT_BIAS.byteSize();
            for (int i = 0; i < biasCount; i++) {
                biasSeg.set(ValueLayout.JAVA_INT, i * stride, biasTokens[i]);
                biasSeg.set(ValueLayout.JAVA_FLOAT, i * stride + 4, biasValues[i]);
            }
            return sampleTokenWithBias(seqId, temperature, repeatPenalty, biasSeg, biasCount);
        }
    }

    /**
     * Samples the next token ID using extended sampler configuration and logit biases from primitive arrays.
     */
    public int sampleTokenWithBias(int seqId, ArgusSamplerConfig config,
                                   int[] biasTokens, float[] biasValues, int biasCount) {
        Objects.requireNonNull(config, "config");
        ArgusValidation.checkNonNegative(biasCount, "biasCount");
        if (biasCount == 0) {
            return sampleToken(seqId, config);
        }
        Objects.requireNonNull(biasTokens, "biasTokens");
        Objects.requireNonNull(biasValues, "biasValues");
        if (biasCount > biasTokens.length || biasCount > biasValues.length) {
            throw new IllegalArgumentException("biasCount exceeds bias array length");
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment biasSeg = localArena.allocate(ArgusLayouts.LOGIT_BIAS, biasCount);
            long stride = ArgusLayouts.LOGIT_BIAS.byteSize();
            for (int i = 0; i < biasCount; i++) {
                biasSeg.set(ValueLayout.JAVA_INT, i * stride, biasTokens[i]);
                biasSeg.set(ValueLayout.JAVA_FLOAT, i * stride + 4, biasValues[i]);
            }
            return sampleTokenWithBias(seqId, config, biasSeg, biasCount);
        }
    }

    /**
     * Primes a sequence slot's sampler history from a primitive int array of tokens.
     */
    public int primeSampler(int seqId, int[] tokens, int nTokens) {
        return primeSampler(seqId, tokens, 0, nTokens);
    }

    /**
     * Primes a sequence slot's sampler history from a slice of a primitive int array of tokens.
     */
    public int primeSampler(int seqId, int[] tokens, int offset, int nTokens) {
        Objects.requireNonNull(tokens, "tokens");
        ArgusValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
        if (nTokens == 0) {
            return 0;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment tokensSeg = localArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment.copy(tokens, offset, tokensSeg, ValueLayout.JAVA_INT, 0, nTokens);
            return primeSampler(seqId, tokensSeg, nTokens);
        }
    }

    /**
     * Synthesizes speech from a Java String into a caller-provided float array.
     */
    public int synthesizeSpeech(ArgusModel wavtokenizerModel, String text, int voiceSeed, float[] outPcm, int maxSamples) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(outPcm, "outPcm");
        ArgusValidation.checkNonNegative(maxSamples, "maxSamples");
        if (maxSamples > outPcm.length) {
            throw new IllegalArgumentException("maxSamples (" + maxSamples + ") exceeds outPcm.length (" + outPcm.length + ")");
        }
        if (maxSamples == 0) {
            return 0;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment textSeg = ArgusValidation.allocateUtf8Text(localArena, text);
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_FLOAT, maxSamples);
            int written = synthesizeSpeech(wavtokenizerModel, textSeg, voiceSeed, outSeg, maxSamples);
            if (written > 0) {
                MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outPcm, 0, written);
            }
            return written;
        }
    }

    /**
     * Synthesizes speech from a Java String and returns a newly allocated float array of the exact sample count.
     */
    public float[] synthesizeSpeech(ArgusModel wavtokenizerModel, String text, int voiceSeed, int maxSamples) {
        Objects.requireNonNull(text, "text");
        ArgusValidation.checkPositive(maxSamples, "maxSamples");
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment textSeg = ArgusValidation.allocateUtf8Text(localArena, text);
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_FLOAT, maxSamples);
            int written = synthesizeSpeech(wavtokenizerModel, textSeg, voiceSeed, outSeg, maxSamples);
            if (written <= 0) {
                return new float[0];
            }
            float[] result = new float[written];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, result, 0, written);
            return result;
        }
    }

    /**
     * Retrieves the embedding vector for a specific sequence ID into a caller-provided float array.
     */
    public int getEmbeddings(int seqId, float[] outEmbeddings, int maxFloats) {
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        ArgusValidation.checkNonNegative(maxFloats, "maxFloats");
        if (maxFloats > outEmbeddings.length) {
            throw new IllegalArgumentException("maxFloats (" + maxFloats + ") exceeds outEmbeddings.length (" + outEmbeddings.length + ")");
        }
        if (maxFloats == 0) {
            return 0;
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_FLOAT, maxFloats);
            int written = getEmbeddings(seqId, outSeg, maxFloats);
            if (written < 0) {
                ArgusNativeException.checkStatus(written, "argus_get_embeddings");
            }
            if (written > 0) {
                MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outEmbeddings, 0, written);
            }
            return written;
        }
    }

    /**
     * Retrieves the embedding vector for sequence 0 into a caller-provided float array.
     */
    public int getEmbeddings(float[] outEmbeddings, int maxFloats) {
        return getEmbeddings(0, outEmbeddings, maxFloats);
    }

    /**
     * Retrieves the embedding vector for a specific sequence ID as a newly allocated float array.
     */
    public float[] getEmbeddings(int seqId) {
        int dim = nEmbd();
        float[] out = new float[dim];
        int written = getEmbeddings(seqId, out, dim);
        if (written != dim) {
            throw new IllegalStateException("Incomplete embedding retrieval: expected " + dim + " floats, got " + written);
        }
        return out;
    }

    /**
     * Retrieves the embedding vector for sequence 0 as a newly allocated float array.
     */
    public float[] getEmbeddings() {
        return getEmbeddings(0);
    }

    /**
     * Evaluates a pre-tokenized int array on a specified sequence slot and retrieves the embedding vector into outEmbeddings.
     */
    public int evalEmbeddings(int[] tokens, int nTokens, int seqId, float[] outEmbeddings) {
        return evalEmbeddings(tokens, 0, nTokens, seqId, outEmbeddings);
    }

    /**
     * Evaluates a slice of a pre-tokenized int array on a specified sequence slot and retrieves the embedding vector into outEmbeddings.
     */
    public int evalEmbeddings(int[] tokens, int offset, int nTokens, int seqId, float[] outEmbeddings) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(outEmbeddings, "outEmbeddings");
        ArgusValidation.checkPositive(nTokens, "nTokens");
        ArgusValidation.checkArraySlice(tokens.length, offset, nTokens, "tokens");
        int requiredDim = nEmbd();
        if (outEmbeddings.length < requiredDim) {
            throw new IllegalArgumentException("outEmbeddings capacity (" + outEmbeddings.length +
                " floats) is insufficient for model embedding dimension (" + requiredDim + " floats)");
        }
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment tokensSeg = localArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment.copy(tokens, offset, tokensSeg, ValueLayout.JAVA_INT, 0, nTokens);
            MemorySegment outSeg = localArena.allocate(ValueLayout.JAVA_FLOAT, outEmbeddings.length);
            int written = evalEmbeddings(tokensSeg, nTokens, seqId, outSeg);
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outEmbeddings, 0, written);
            return written;
        }
    }

    /**
     * Evaluates a pre-tokenized int array on sequence slot 0 and retrieves the embedding vector into outEmbeddings.
     */
    public int evalEmbeddings(int[] tokens, int nTokens, float[] outEmbeddings) {
        return evalEmbeddings(tokens, nTokens, 0, outEmbeddings);
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming.
     *
     * @param prompt    input text prompt
     * @param maxTokens maximum number of tokens to generate
     * @param config    sampler configuration
     * @param onPiece   optional streaming callback invoked for each decoded UTF-8 text chunk
     * @return complete generated text string
     */
    public String generate(String prompt, int maxTokens, ArgusSamplerConfig config, Consumer<String> onPiece) {
        return generate(prompt, maxTokens, config, onPiece, null);
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming
     * and optional cooperative cancellation.
     *
     * @param prompt    input text prompt
     * @param maxTokens maximum number of tokens to generate
     * @param config    sampler configuration
     * @param onPiece   optional streaming callback invoked for each decoded UTF-8 text chunk
     * @param abortFlag optional cancellation flag
     * @return complete generated text string
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
     *
     * @param prompt         input text prompt
     * @param maxTokens      maximum number of tokens to generate
     * @param config         sampler configuration
     * @param shouldContinue optional streaming predicate invoked for each decoded UTF-8 text chunk (return {@code false} to stop)
     * @return complete generated text string
     */
    public String generateUntil(String prompt, int maxTokens, ArgusSamplerConfig config, Predicate<String> shouldContinue) {
        return generateUntil(prompt, maxTokens, config, shouldContinue, null);
    }

    /**
     * High-level convenience method to generate text from a prompt using stateful multi-byte UTF-8 streaming,
     * an early-stopping predicate callback, and optional cooperative cancellation.
     *
     * @param prompt         input text prompt
     * @param maxTokens      maximum number of tokens to generate
     * @param config         sampler configuration
     * @param shouldContinue optional streaming predicate invoked for each decoded UTF-8 text chunk (return {@code false} to stop)
     * @param abortFlag      optional cancellation flag
     * @return complete generated text string
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
