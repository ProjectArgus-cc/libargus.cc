package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Represents an active text inference and speech generation session context.
 * Owns a private shared arena for concurrent thread safety.
 * Extends {@link ArgusNativeResource} for thread-safe handle leasing and idempotent lifecycle management.
 */
public final class ArgusContext extends ArgusNativeResource {
    private final ArgusModel modelRef;
    private final ArgusModel draftModelRef;
    private final Arena contextArena;
    private final MemorySegment batchSeg;
    private final MemorySegment samplerParamsSeg;
    private final ReentrantLock batchLock = new ReentrantLock();
    private final ReentrantLock samplerLock = new ReentrantLock();
    private final ReentrantLock ttsLock = new ReentrantLock();

    private MemorySegment ttsWorkspace = MemorySegment.NULL;
    private long ttsWorkspaceSizeFloats = 0;

    MemorySegment getTtsWorkspace() {
        return ttsWorkspace;
    }

    long getTtsWorkspaceSizeFloats() {
        return ttsWorkspaceSizeFloats;
    }

    private ArgusContext(MemorySegment ctxPtr, ArgusModel modelRef, ArgusModel draftModelRef, Arena contextArena) {
        super(ctxPtr);
        this.modelRef = Objects.requireNonNull(modelRef);
        this.draftModelRef = draftModelRef;
        this.contextArena = Objects.requireNonNull(contextArena);
        this.batchSeg = contextArena.allocate(ArgusLayouts.TOKEN_BATCH);
        this.samplerParamsSeg = contextArena.allocate(ArgusLayouts.SAMPLER_PARAMS);
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
        MemorySegment modelH = model.acquireReadLease();
        try {
            MemorySegment draftModelHandle = MemorySegment.NULL;
            if (config.draftModel() != null) {
                draftModelHandle = config.draftModel().acquireReadLease();
            }
            try {
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

                MemorySegment ctxPtr = (MemorySegment) ArgusBindings.argus_context_init.invokeExact(modelH, paramsSeg);
                if (ctxPtr.equals(MemorySegment.NULL)) {
                    ArgusNativeException.checkStatus(-1, "argus_context_init");
                }
                return new ArgusContext(ctxPtr, model, config.draftModel(), privateArena);
            } finally {
                if (config.draftModel() != null) {
                    config.draftModel().releaseReadLease();
                }
            }
        } catch (Throwable t) {
            privateArena.close();
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to initialize native context", t);
        } finally {
            model.releaseReadLease();
        }
    }

    /**
     * Converts raw text into vocabulary token IDs.
     * This operation is lock-free and read-only.
     */
    public int tokenize(MemorySegment textSeg, MemorySegment outTokensSeg, boolean addBos) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outTokensSeg);
        ArgusValidation.checkReadable(textSeg, 0, "textSeg");
        ArgusValidation.checkWritable(outTokensSeg, ValueLayout.JAVA_INT.byteSize(), "outTokensSeg");
        long maxTokens = outTokensSeg.byteSize() / ValueLayout.JAVA_INT.byteSize();
        long textLen = textSeg.byteSize();

        MemorySegment modelH = modelRef.acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_tokenize_n.invokeExact(
                modelH,
                textSeg,
                textLen,
                outTokensSeg,
                (int) maxTokens,
                addBos
            );
            if (res < 0) {
                ArgusNativeException.checkStatus(res, "argus_tokenize_n");
            }
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to tokenize input text", t);
        } finally {
            modelRef.releaseReadLease();
        }
    }

    /**
     * Converts a single token ID to its text representation (piece).
     * This operation is lock-free and read-only.
     */
    public String tokenToPiece(int token) {
        MemorySegment modelH = modelRef.acquireReadLease();
        try (Arena local = Arena.ofConfined()) {
            int bufSize = 256;
            MemorySegment bufSeg = local.allocate(bufSize);

            int result = (int) ArgusBindings.argus_token_to_piece.invokeExact(
                modelH,
                token,
                bufSeg,
                bufSize
            );

            if (result < 0) {
                return "";
            }

            byte[] bytes = new byte[result];
            MemorySegment.copy(bufSeg, ValueLayout.JAVA_BYTE, 0, 
                MemorySegment.ofArray(bytes), ValueLayout.JAVA_BYTE, 0, result);
            return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to convert token " + token + " to piece", t);
        } finally {
            modelRef.releaseReadLease();
        }
    }

    /**
     * Synchronously decodes a batch of tokens in the context.
     * Mutex-locked inside the native layer to prevent concurrency issues.
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
        MemorySegment abortFlagSeg = MemorySegment.NULL;
        if (abortFlag != null) {
            abortFlagSeg = abortFlag.acquireReadLease();
        }
        try {
            return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, abortFlagSeg);
        } finally {
            if (abortFlag != null) {
                abortFlag.releaseReadLease();
            }
        }
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
     * @param abortFlagSeg  optional memory segment (pointer targeting argus_abort_flag_t) for cancellation
     * @return 0 on success, -2 if aborted, non-zero on failure
     */
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits, MemorySegment abortFlagSeg) {
        Objects.requireNonNull(tokensSeg);
        Objects.requireNonNull(abortFlagSeg);
        if (nTokens == 0) {
            return 0;
        }

        long requiredBytes = ArgusValidation.multiplyExactBytes(nTokens, ValueLayout.JAVA_INT.byteSize(), "tokensSeg");
        ArgusValidation.checkReadable(tokensSeg, requiredBytes, "tokensSeg");
        ArgusValidation.checkNonNegative(startPos, "startPos");
        ArgusValidation.checkNonNegative(seqId, "seqId");

        MemorySegment ctxH = acquireReadLease();
        try {
            batchLock.lock();
            try {
                batchSeg.set(ValueLayout.ADDRESS, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("tokens")), 
                    tokensSeg
                );
                batchSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")), 
                    nTokens
                );
                batchSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("start_pos")), 
                    startPos
                );
                batchSeg.set(ValueLayout.JAVA_INT, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("seq_id")), 
                    seqId
                );
                batchSeg.set(ValueLayout.JAVA_BOOLEAN, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("request_logits")), 
                    requestLogits
                );
                batchSeg.set(ValueLayout.ADDRESS, 
                    ArgusLayouts.TOKEN_BATCH.byteOffset(MemoryLayout.PathElement.groupElement("abort_flag")), 
                    abortFlagSeg
                );

                return (int) ArgusBindings.argus_decode_batch.invokeExact(ctxH, batchSeg);
            } finally {
                batchLock.unlock();
            }
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
            MemorySegment mctxH = mctx.acquireReadLease();
            try {
                MemorySegment chunksH = chunks.acquireReadLease();
                try (Arena local = Arena.ofConfined()) {
                    MemorySegment outNewNPast = local.allocate(ValueLayout.JAVA_INT);
                    int res = (int) ArgusBindings.argus_eval_multimodal_chunks.invokeExact(
                        mctxH,
                        ctxH,
                        chunksH,
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
                } finally {
                    chunks.releaseReadLease();
                }
            } finally {
                mctx.releaseReadLease();
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
     */
    public void clearCacheSlot(int seqId, int p0, int p1) {
        MemorySegment ctxH = acquireReadLease();
        try {
            ArgusBindings.argus_kv_cache_clear_slot.invokeExact(ctxH, seqId, p0, p1);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to prune KV cache sequence slot", t);
        } finally {
            releaseReadLease();
        }
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
            MemorySegment wavModelHandle = MemorySegment.NULL;
            if (wavtokenizerModel != null) {
                wavModelHandle = wavtokenizerModel.acquireReadLease();
            }
            try {
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
                if (wavtokenizerModel != null) {
                    wavtokenizerModel.releaseReadLease();
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
}
