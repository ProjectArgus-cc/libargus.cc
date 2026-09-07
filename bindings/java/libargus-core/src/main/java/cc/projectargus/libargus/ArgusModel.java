package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import cc.projectargus.libargus.internal.ArgusNativeResource;
import cc.projectargus.libargus.internal.ArgusValidation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wraps loaded GGUF model weights in unmanaged memory.
 * Extends {@link ArgusNativeResource} for thread-safe handle leasing and idempotent lifecycle management.
 */
public final class ArgusModel extends ArgusNativeResource {

    ArgusModel(MemorySegment modelPtr) {
        super(modelPtr);
    }

    @Override
    protected String resourceName() {
        return "ArgusModel";
    }

    @Override
    protected void releaseNative(MemorySegment oldHandle) {
        try {
            ArgusBindings.argus_model_release.invokeExact(oldHandle);
        } catch (Throwable t) {
            // Suppress destruction exceptions
        }
    }

    void clearHandleForTesting() {
        this.handle = MemorySegment.NULL;
    }

    /**
     * Loads GGUF weights off-heap using the specified configuration parameters.
     *
     * @param arena     the allocation scope for the temporary parameter structures
     * @param modelPath path to the local GGUF file
     * @param gpuLayers number of layers to offload to VRAM (RTX 4090 etc.)
     * @param useMlock  force locking pages to RAM to prevent swapping
     * @return a loaded ArgusModel wrapper
     */
    public static ArgusModel load(Arena arena, Path modelPath, int gpuLayers, boolean useMlock) {
        Objects.requireNonNull(arena);
        Objects.requireNonNull(modelPath);

        // Allocate local path string off-heap
        MemorySegment pathSegment = arena.allocateFrom(modelPath.toAbsolutePath().toString());

        // Allocate the argus_model_params struct off-heap
        MemorySegment paramsSegment = arena.allocate(ArgusLayouts.MODEL_PARAMS);

        // Populate struct fields using offset layout mappings
        paramsSegment.set(ValueLayout.ADDRESS, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("model_path")), 
            pathSegment
        );
        paramsSegment.set(ValueLayout.JAVA_INT, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("gpu_layers")), 
            gpuLayers
        );
        paramsSegment.set(ValueLayout.JAVA_BOOLEAN, 
            ArgusLayouts.MODEL_PARAMS.byteOffset(MemoryLayout.PathElement.groupElement("use_mlock")), 
            useMlock
        );

        try {
            MemorySegment modelPtr = (MemorySegment) ArgusBindings.argus_model_load.invokeExact(paramsSegment);
            if (modelPtr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Native argus_model_load returned NULL for: " + modelPath);
            }
            return new ArgusModel(modelPtr);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to execute native load for " + modelPath, t);
        }
    }

    /**
     * Returns the Beginning-Of-Sentence (BOS) token ID.
     */
    public int vocabBos() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_vocab_bos.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve BOS token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the End-Of-Sentence (EOS) token ID.
     */
    public int vocabEos() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_vocab_eos.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve EOS token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the End-Of-Turn (EOT) token ID.
     */
    public int vocabEot() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_vocab_eot.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve EOT token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the Padding (PAD) token ID.
     */
    public int vocabPad() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_vocab_pad.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve PAD token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the total vocabulary token size count.
     */
    public int vocabNTokens() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_vocab_n_tokens.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve vocabulary size", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Checks if the given token ID is an End-Of-Generation (EOG) token.
     */
    public boolean vocabIsEog(int token) {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_vocab_is_eog.invokeExact(h, token);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to check EOG token: " + token, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Extracts model GGUF metadata string values by key name.
     * Uses caller-provided buffers for absolute zero-allocation lookup.
     *
     * @param keySegment  off-heap address containing null-terminated UTF-8 key string
     * @param valueBuffer destination off-heap character array segment
     * @param bufferSize  maximum capacity limit of target buffer segment
     * @return character length successfully written, or negative on failure.
     */
    public int getMetadataValue(MemorySegment keySegment, MemorySegment valueBuffer, int bufferSize) {
        Objects.requireNonNull(keySegment);
        Objects.requireNonNull(valueBuffer);
        ArgusValidation.checkPositive(bufferSize, "bufferSize");
        ArgusValidation.checkWritable(valueBuffer, bufferSize, "valueBuffer");
        long keyLen = 0;
        if (keySegment.byteSize() > 0) {
            while (keyLen < keySegment.byteSize() && keySegment.get(ValueLayout.JAVA_BYTE, keyLen) != 0) {
                keyLen++;
            }
        }
        return getMetadataValue(keySegment, keyLen, valueBuffer, bufferSize);
    }

    /**
     * Extracts model GGUF metadata string values by key name with explicit length.
     * Uses caller-provided buffers for absolute zero-allocation lookup.
     *
     * @param keySegment  off-heap address containing UTF-8 key string
     * @param keyLen      exact length in bytes of the key string
     * @param valueBuffer destination off-heap character array segment
     * @param bufferSize  maximum capacity limit of target buffer segment
     * @return character length successfully written, or negative on failure.
     */
    public int getMetadataValue(MemorySegment keySegment, long keyLen, MemorySegment valueBuffer, int bufferSize) {
        Objects.requireNonNull(keySegment);
        Objects.requireNonNull(valueBuffer);
        ArgusValidation.checkPositive(bufferSize, "bufferSize");
        ArgusValidation.checkWritable(valueBuffer, bufferSize, "valueBuffer");
        ArgusValidation.checkNonNegative(keyLen, "keyLen");
        if (keyLen > 0) {
            ArgusValidation.checkReadable(keySegment, keyLen, "keySegment");
        }
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_meta_val_str_n.invokeExact(h, keySegment, keyLen, valueBuffer, bufferSize);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve metadata string", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Extracts model GGUF metadata string values by key name.
     *
     * @param key metadata key name
     * @return metadata value string, or null if key is missing or failed
     */
    public String getMetadataValue(String key) {
        Objects.requireNonNull(key);
        MemorySegment h = acquireReadLease();
        try (Arena localArena = Arena.ofConfined()) {
            byte[] keyBytes = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            MemorySegment keySeg = localArena.allocate(keyBytes.length + 1);
            MemorySegment.copy(MemorySegment.ofArray(keyBytes), 0, keySeg, 0, keyBytes.length);
            keySeg.set(ValueLayout.JAVA_BYTE, keyBytes.length, (byte) 0);
            long keyLen = keyBytes.length;

            int initialSize = 512;
            MemorySegment valSeg = localArena.allocate(initialSize);
            int len = (int) ArgusBindings.argus_model_meta_val_str_n.invokeExact(h, keySeg, keyLen, valSeg, initialSize);
            if (len < 0) {
                return null;
            }
            if (len >= initialSize) {
                MemorySegment largerSeg = localArena.allocate(len + 1);
                len = (int) ArgusBindings.argus_model_meta_val_str_n.invokeExact(h, keySeg, keyLen, largerSeg, len + 1);
                if (len < 0) {
                    return null;
                }
                return largerSeg.getString(0);
            }
            return valSeg.getString(0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve metadata string for key: " + key, t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Traverses and retrieves all GGUF metadata key-value pairs stored in the model.
     *
     * @return a map containing the model's metadata dictionary
     */
    public java.util.Map<String, String> getMetadataMap() {
        java.util.Map<String, String> map = new java.util.LinkedHashMap<>();
        MemorySegment h = acquireReadLease();
        try {
            int count = (int) ArgusBindings.argus_model_meta_count.invokeExact(h);
            if (count < 0) {
                return map;
            }
            try (Arena localArena = Arena.ofConfined()) {
                int bufferSize = 1024;
                MemorySegment keyBuffer = localArena.allocate(bufferSize);
                MemorySegment valBuffer = localArena.allocate(bufferSize);
                for (int i = 0; i < count; i++) {
                    int keyLen = (int) ArgusBindings.argus_model_meta_key_by_index.invokeExact(h, i, keyBuffer, bufferSize);
                    int valLen = (int) ArgusBindings.argus_model_meta_val_str_by_index.invokeExact(h, i, valBuffer, bufferSize);
                    
                    MemorySegment keySeg = keyBuffer;
                    if (keyLen >= bufferSize) {
                        keySeg = localArena.allocate(keyLen + 1);
                        ArgusBindings.argus_model_meta_key_by_index.invokeExact(h, i, keySeg, keyLen + 1);
                    }
                    MemorySegment valSeg = valBuffer;
                    if (valLen >= bufferSize) {
                        valSeg = localArena.allocate(valLen + 1);
                        ArgusBindings.argus_model_meta_val_str_by_index.invokeExact(h, i, valSeg, valLen + 1);
                    }
                    
                    if (keyLen >= 0 && valLen >= 0) {
                        map.put(keySeg.getString(0), valSeg.getString(0));
                    }
                }
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve model metadata map", t);
        } finally {
            releaseReadLease();
        }
        return map;
    }

    /**
     * Returns the model's embedding dimension.
     */
    public int nEmbd() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_embd.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve embedding dimension", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the model's training context length limit.
     */
    public int nCtxTrain() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_ctx_train.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve training context limit", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the model's transformer layer count.
     */
    public int nLayer() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_layer.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve layer count", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the model's attention query head count.
     */
    public int nHead() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_head.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve query head count", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the model's attention key-value head count.
     */
    public int nHeadKv() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_head_kv.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve key-value head count", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the model's total parameter count.
     */
    public long nParams() {
        MemorySegment h = acquireReadLease();
        try {
            return (long) ArgusBindings.argus_model_n_params.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to retrieve parameter count", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Checks if the model architecture contains an encoder stack or non-causal topology.
     */
    public boolean hasEncoder() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_model_has_encoder.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model encoder topology", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the rotary position embedding dimensions per token (4 for M-RoPE, 1 for standard 1D-RoPE).
     */
    public int nPosPerEmbd() {
        MemorySegment h = acquireReadLease();
        try {
            return (int) ArgusBindings.argus_model_n_pos_per_embd.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model n_pos_per_embd", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Checks whether the loaded model uses Multimodal Rotary Position Embeddings (M-RoPE / IM-RoPE).
     */
    public boolean isMRoPE() {
        MemorySegment h = acquireReadLease();
        try {
            return (boolean) ArgusBindings.argus_model_is_mrope.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model is_mrope", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns the total memory size of model weights in bytes.
     */
    public long modelSize() {
        MemorySegment h = acquireReadLease();
        try {
            return (long) ArgusBindings.argus_model_size.invokeExact(h);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model size", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns a human-readable string describing the model architecture.
     */
    public String desc() {
        MemorySegment h = acquireReadLease();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(128);
            int len = (int) ArgusBindings.argus_model_desc.invokeExact(h, buf, 128);
            if (len < 0) return "";
            return buf.getString(0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query model description", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Calculates the KV cache memory footprint in bytes per token for the entire layer stack.
     */
    public long kvBytesPerToken(int typeK, int typeV) {
        MemorySegment h = acquireReadLease();
        try {
            return (long) ArgusBindings.argus_model_kv_bytes_per_token.invokeExact(h, typeK, typeV);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to calculate KV bytes per token", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Estimates total memory requirement (Weights + KV Cache) for a target context length.
     */
    public long estimateVramBytes(int contextLength, int typeK, int typeV) {
        MemorySegment h = acquireReadLease();
        try {
            return (long) ArgusBindings.argus_model_estimate_vram_bytes.invokeExact(h, contextLength, typeK, typeV);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to estimate VRAM bytes", t);
        } finally {
            releaseReadLease();
        }
    }

    /**
     * Returns block size in bytes for the specified GGML quantization type.
     */
    public static long quantTypeSize(int type) {
        try {
            return (long) ArgusBindings.argus_quant_type_size.invokeExact(type);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query GGML quantization type size", t);
        }
    }

    /**
     * Returns element count per block for the specified GGML quantization type.
     */
    public static int quantBlockSize(int type) {
        try {
            return (int) ArgusBindings.argus_quant_block_size.invokeExact(type);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to query GGML quantization block size", t);
        }
    }

    /**
     * Converts text into vocabulary token IDs directly using this model's unmanaged vocabulary.
     * Thread-safe, lock-free, zero-copy.
     *
     * @param textSeg      input UTF-8 text segment
     * @param outTokensSeg destination int32 token array segment
     * @param addBos       whether to prepend BOS token
     * @return count of tokens written
     */
    public int tokenize(MemorySegment textSeg, MemorySegment outTokensSeg, boolean addBos) {
        Objects.requireNonNull(textSeg);
        Objects.requireNonNull(outTokensSeg);
        ArgusValidation.checkReadable(textSeg, 0, "textSeg");
        ArgusValidation.checkWritable(outTokensSeg, ValueLayout.JAVA_INT.byteSize(), "outTokensSeg");
        long maxTokens = outTokensSeg.byteSize() / ValueLayout.JAVA_INT.byteSize();
        long textLen = textSeg.byteSize();

        MemorySegment h = acquireReadLease();
        try {
            int res = (int) ArgusBindings.argus_tokenize_n.invokeExact(
                h,
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
            releaseReadLease();
        }
    }
}
