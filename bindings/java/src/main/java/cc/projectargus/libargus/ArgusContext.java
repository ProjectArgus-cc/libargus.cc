package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Represents an active text inference and speech generation session context.
 * Implements AutoCloseable to ensure native resources are safely released.
 */
public final class ArgusContext implements AutoCloseable {
    private MemorySegment ctxPtr;
    private final ArgusModel modelRef;
    private final ArgusModel draftModelRef;
    private final MemorySegment batchSeg;
    private final Arena arena;

    private MemorySegment ttsWorkspace = MemorySegment.NULL;
    private long ttsWorkspaceSizeFloats = 0;

    MemorySegment getTtsWorkspace() {
        return ttsWorkspace;
    }

    long getTtsWorkspaceSizeFloats() {
        return ttsWorkspaceSizeFloats;
    }

    private ArgusContext(MemorySegment ctxPtr, ArgusModel modelRef, ArgusModel draftModelRef, Arena arena) {
        this.ctxPtr = Objects.requireNonNull(ctxPtr);
        this.modelRef = Objects.requireNonNull(modelRef);
        this.draftModelRef = draftModelRef;
        this.arena = Objects.requireNonNull(arena);
        this.batchSeg = arena.allocate(ArgusLayouts.TOKEN_BATCH);
    }

    /**
     * Initializes an execution context on the specified model weights.
     *
     * @param arena  the allocation scope for context parameter configuration
     * @param model  the loaded model weights
     * @param config session parameters (length, threads, cache formats, etc.)
     * @return an active ArgusContext
     */
    public static ArgusContext init(Arena arena, ArgusModel model, ArgusContextConfig config) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(model);
        Objects.requireNonNull(config);

        model.acquire();
        if (config.draftModel() != null) {
            config.draftModel().acquire();
        }

        try {
            MemorySegment paramsSeg = arena.allocate(ArgusLayouts.CONTEXT_PARAMS);

            MemorySegment draftModelHandle = (config.draftModel() != null) 
                ? config.draftModel().getHandle() 
                : MemorySegment.NULL;

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
            paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("enable_draft_mtp")), 
                config.enableDraftMtp()
            );
            paramsSeg.set(ValueLayout.JAVA_BOOLEAN, 
                ArgusLayouts.CONTEXT_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("embeddings")), 
                config.embeddings()
            );

            MemorySegment ctxPtr = (MemorySegment) ArgusBindings.argus_context_init.invokeExact(model.getHandle(), paramsSeg);
            if (ctxPtr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_context_init returned NULL");
            }
            return new ArgusContext(ctxPtr, model, config.draftModel(), arena);
        } catch (Throwable t) {
            model.release();
            if (config.draftModel() != null) {
                config.draftModel().release();
            }
            throw new RuntimeException("Failed to initialize native context", t);
        }
    }

    /**
     * Converts raw text into vocabulary token IDs.
     * This operation is lock-free and read-only.
     */
    public int tokenize(MemorySegment textSeg, MemorySegment outTokensSeg, boolean addBos) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outTokensSeg);
        long nTokens = outTokensSeg.byteSize() / ValueLayout.JAVA_INT.byteSize();
        if (nTokens == 0) {
            return 0;
        }

        try {
            return (int) ArgusBindings.argus_tokenize.invokeExact(
                modelRef.getHandle(),
                textSeg,
                outTokensSeg,
                (int) nTokens,
                addBos
            );
        } catch (Throwable t) {
            throw new RuntimeException("Failed to tokenize input text", t);
        }
    }

    /**
     * Converts a single token ID to its text representation (piece).
     * This operation is lock-free and read-only.
     */
    public String tokenToPiece(int token) {
        try (Arena local = Arena.ofConfined()) {
            int bufSize = 256;
            MemorySegment bufSeg = local.allocate(bufSize);

            int result = (int) ArgusBindings.argus_token_to_piece.invokeExact(
                modelRef.getHandle(),
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
            throw new RuntimeException("Failed to convert token " + token + " to piece", t);
        }
    }

    /**
     * Synchronously decodes a batch of tokens in the context.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     *
     * @param tokens        tokens to evaluate
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
     * @param tokens        tokens to evaluate
     * @param startPos      KV cache offset position
     * @param seqId         sequence tracking ID
     * @param requestLogits true to evaluate logits on the final token of the batch
     * @param abortFlag     optional safe cancellation flag object
     * @return 0 on success, -2 if aborted, non-zero on failure
     */
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits, ArgusAbortFlag abortFlag) {
        MemorySegment abortFlagSeg = (abortFlag != null) ? abortFlag.getHandle() : MemorySegment.NULL;
        return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, abortFlagSeg);
    }

    /**
     * Synchronously decodes a batch of tokens in the context with optional early cancellation.
     * Mutex-locked inside the native layer to prevent concurrency issues.
     *
     * @param tokens        tokens to evaluate
     * @param startPos      KV cache offset position
     * @param seqId         sequence tracking ID
     * @param requestLogits true to evaluate logits on the final token of the batch
     * @param abortFlagSeg  optional memory segment (pointer targeting 4-byte int) for cancellation
     * @return 0 on success, -2 if aborted, non-zero on failure
     */
    public int decodeBatch(MemorySegment tokensSeg, int nTokens, int startPos, int seqId, boolean requestLogits, MemorySegment abortFlagSeg) {
        Objects.requireNonNull(tokensSeg);
        Objects.requireNonNull(abortFlagSeg);
        if (nTokens == 0) {
            return 0;
        }

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

            return (int) ArgusBindings.argus_decode_batch.invokeExact(ctxPtr, batchSeg);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to decode batch of tokens", t);
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

        try (Arena local = Arena.ofConfined()) {
            MemorySegment outNewNPast = local.allocate(ValueLayout.JAVA_INT);
            int res = (int) ArgusBindings.argus_eval_multimodal_chunks.invokeExact(
                mctx.getHandle(),
                ctxPtr,
                chunks.getHandle(),
                nPast,
                seqId,
                nBatch,
                logitsLast,
                outNewNPast
            );

            if (res != 0) {
                throw new RuntimeException("Native argus_eval_multimodal_chunks returned error code: " + res);
            }

            return outNewNPast.get(ValueLayout.JAVA_INT, 0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to evaluate multimodal chunks", t);
        }
    }


    /**
     * Samples the next token ID from the last computed logits matrix.
     * Mutex-locked inside the native layer.
     */
    public int sampleToken(int seqId, float temperature, float repeatPenalty) {
        try {
            return (int) ArgusBindings.argus_sample_token.invokeExact(ctxPtr, seqId, temperature, repeatPenalty);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to sample token", t);
        }
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
        try {
            return (int) ArgusBindings.argus_sample_token_with_bias.invokeExact(
                ctxPtr, seqId, temperature, repeatPenalty, biasSegment, biasCount
            );
        } catch (Throwable t) {
            throw new RuntimeException("Failed to sample token with bias", t);
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
        try {
            ArgusBindings.argus_kv_cache_clear_slot.invokeExact(ctxPtr, seqId, p0, p1);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to prune KV cache sequence slot", t);
        }
    }

    /**
     * Synthesizes audio samples using WavTokenizer and OuteTTS pipelines.
     * Mutex-locked inside the native layer.
     *
     * @param wavtokenizerModel GGUF weights of the WavTokenizer vocoder model
     * @param text               input text prompt
     * @param voiceSeed          seed controlling speaker characteristics
     * @param outPcm             destination buffer to write 24kHz float PCM audio into
     * @return sample count generated, or negative on failure
     */
    public int synthesizeSpeech(ArgusModel wavtokenizerModel, MemorySegment textSeg, int voiceSeed, MemorySegment outPcmSeg, int maxSamples) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outPcmSeg);
        if (maxSamples <= 0) {
            return 0;
        }

        try {
            MemorySegment wavModelHandle = (wavtokenizerModel != null) ? wavtokenizerModel.getHandle() : MemorySegment.NULL;

            int res = (int) ArgusBindings.argus_synthesize_speech.invokeExact(
                ctxPtr,
                wavModelHandle,
                textSeg,
                voiceSeed,
                outPcmSeg,
                maxSamples,
                ttsWorkspace,
                ttsWorkspaceSizeFloats
            );

            if (res < -1) {
                // Resize off-heap workspace buffer
                long requiredFloats = -res;
                this.ttsWorkspace = arena.allocate(
                    requiredFloats * ValueLayout.JAVA_FLOAT.byteSize(),
                    ValueLayout.JAVA_FLOAT.byteAlignment()
                );
                this.ttsWorkspaceSizeFloats = requiredFloats;

                // Retry execution
                res = (int) ArgusBindings.argus_synthesize_speech.invokeExact(
                    ctxPtr,
                    wavModelHandle,
                    textSeg,
                    voiceSeed,
                    outPcmSeg,
                    maxSamples,
                    ttsWorkspace,
                    ttsWorkspaceSizeFloats
                );
            }

            return res;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to synthesize speech", t);
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
        if (maxFloats <= 0) {
            return 0;
        }
        try {
            return (int) ArgusBindings.argus_get_embeddings.invokeExact(ctxPtr, seqId, outEmbeddings, maxFloats);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve embeddings from native context", t);
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
                ArgusBindings.argus_context_free.invokeExact(ctxPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to release native context resources", t);
            } finally {
                ctxPtr = MemorySegment.NULL;
                ttsWorkspace = MemorySegment.NULL;
                ttsWorkspaceSizeFloats = 0;
                modelRef.release();
                if (draftModelRef != null) {
                    draftModelRef.release();
                }
            }
        }
    }
}
