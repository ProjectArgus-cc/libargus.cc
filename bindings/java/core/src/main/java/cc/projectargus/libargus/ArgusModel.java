package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusLayouts;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wraps loaded GGUF model weights in unmanaged memory.
 * Implements AutoCloseable to ensure resources are cleaned up safely.
 */
public final class ArgusModel implements AutoCloseable {
    private MemorySegment modelPtr;
    private final java.util.concurrent.atomic.AtomicInteger refCount = new java.util.concurrent.atomic.AtomicInteger(1);

    ArgusModel(MemorySegment modelPtr) {
        this.modelPtr = Objects.requireNonNull(modelPtr);
    }

    void acquire() {
        int current;
        do {
            current = refCount.get();
            if (current <= 0) {
                throw new IllegalStateException("Cannot acquire reference; ArgusModel has already been closed");
            }
        } while (!refCount.compareAndSet(current, current + 1));
    }

    void release() {
        if (refCount.decrementAndGet() == 0) {
            freeNativeModel();
        }
    }

    int getRefCount() {
        return refCount.get();
    }

    void clearHandleForTesting() {
        this.modelPtr = MemorySegment.NULL;
    }

    private synchronized void freeNativeModel() {
        if (modelPtr != null && !modelPtr.equals(MemorySegment.NULL)) {
            try {
                ArgusBindings.argus_model_free.invokeExact(modelPtr);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to free native model resources", t);
            } finally {
                modelPtr = MemorySegment.NULL;
            }
        }
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
            throw new RuntimeException("Failed to execute native load for " + modelPath, t);
        }
    }

    /**
     * Returns the Beginning-Of-Sentence (BOS) token ID.
     */
    public int vocabBos() {
        try {
            return (int) ArgusBindings.argus_vocab_bos.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve BOS token", t);
        }
    }

    /**
     * Returns the End-Of-Sentence (EOS) token ID.
     */
    public int vocabEos() {
        try {
            return (int) ArgusBindings.argus_vocab_eos.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve EOS token", t);
        }
    }

    /**
     * Returns the End-Of-Turn (EOT) token ID.
     */
    public int vocabEot() {
        try {
            return (int) ArgusBindings.argus_vocab_eot.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve EOT token", t);
        }
    }

    /**
     * Returns the Padding (PAD) token ID.
     */
    public int vocabPad() {
        try {
            return (int) ArgusBindings.argus_vocab_pad.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve PAD token", t);
        }
    }

    /**
     * Returns the total vocabulary token size count.
     */
    public int vocabNTokens() {
        try {
            return (int) ArgusBindings.argus_vocab_n_tokens.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve vocabulary size", t);
        }
    }

    /**
     * Checks if the given token ID is an End-Of-Generation (EOG) token.
     */
    public boolean vocabIsEog(int token) {
        try {
            return (boolean) ArgusBindings.argus_vocab_is_eog.invokeExact(modelPtr, token);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to check EOG token: " + token, t);
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
        try {
            return (int) ArgusBindings.argus_model_meta_val_str.invokeExact(modelPtr, keySegment, valueBuffer, bufferSize);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve metadata string", t);
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
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment keySeg = localArena.allocateFrom(key);
            int initialSize = 512;
            MemorySegment valSeg = localArena.allocate(initialSize);
            int len = (int) ArgusBindings.argus_model_meta_val_str.invokeExact(modelPtr, keySeg, valSeg, initialSize);
            if (len < 0) {
                return null;
            }
            if (len >= initialSize) {
                MemorySegment largerSeg = localArena.allocate(len + 1);
                len = (int) ArgusBindings.argus_model_meta_val_str.invokeExact(modelPtr, keySeg, largerSeg, len + 1);
                if (len < 0) {
                    return null;
                }
                return largerSeg.getString(0);
            }
            return valSeg.getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve metadata string for key: " + key, t);
        }
    }

    /**
     * Traverses and retrieves all GGUF metadata key-value pairs stored in the model.
     *
     * @return a map containing the model's metadata dictionary
     */
    public java.util.Map<String, String> getMetadataMap() {
        java.util.Map<String, String> map = new java.util.LinkedHashMap<>();
        try {
            int count = (int) ArgusBindings.argus_model_meta_count.invokeExact(modelPtr);
            if (count < 0) {
                return map;
            }
            try (Arena localArena = Arena.ofConfined()) {
                int bufferSize = 1024;
                MemorySegment keyBuffer = localArena.allocate(bufferSize);
                MemorySegment valBuffer = localArena.allocate(bufferSize);
                for (int i = 0; i < count; i++) {
                    int keyLen = (int) ArgusBindings.argus_model_meta_key_by_index.invokeExact(modelPtr, i, keyBuffer, bufferSize);
                    int valLen = (int) ArgusBindings.argus_model_meta_val_str_by_index.invokeExact(modelPtr, i, valBuffer, bufferSize);
                    
                    MemorySegment keySeg = keyBuffer;
                    if (keyLen >= bufferSize) {
                        keySeg = localArena.allocate(keyLen + 1);
                        ArgusBindings.argus_model_meta_key_by_index.invokeExact(modelPtr, i, keySeg, keyLen + 1);
                    }
                    MemorySegment valSeg = valBuffer;
                    if (valLen >= bufferSize) {
                        valSeg = localArena.allocate(valLen + 1);
                        ArgusBindings.argus_model_meta_val_str_by_index.invokeExact(modelPtr, i, valSeg, valLen + 1);
                    }
                    
                    if (keyLen >= 0 && valLen >= 0) {
                        map.put(keySeg.getString(0), valSeg.getString(0));
                    }
                }
            }
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve model metadata map", t);
        }
        return map;
    }

    /**
     * Returns the model's embedding dimension.
     */
    public int nEmbd() {
        try {
            return (int) ArgusBindings.argus_model_n_embd.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve embedding dimension", t);
        }
    }

    /**
     * Returns the model's training context length limit.
     */
    public int nCtxTrain() {
        try {
            return (int) ArgusBindings.argus_model_n_ctx_train.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve training context limit", t);
        }
    }

    /**
     * Returns the model's transformer layer count.
     */
    public int nLayer() {
        try {
            return (int) ArgusBindings.argus_model_n_layer.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve layer count", t);
        }
    }

    /**
     * Returns the model's attention query head count.
     */
    public int nHead() {
        try {
            return (int) ArgusBindings.argus_model_n_head.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve query head count", t);
        }
    }

    /**
     * Returns the model's attention key-value head count.
     */
    public int nHeadKv() {
        try {
            return (int) ArgusBindings.argus_model_n_head_kv.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve key-value head count", t);
        }
    }

    /**
     * Returns the model's total parameter count.
     */
    public long nParams() {
        try {
            return (long) ArgusBindings.argus_model_n_params.invokeExact(modelPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve parameter count", t);
        }
    }

    /**
     * Returns the raw memory address representing the unmanaged model structure.
     */
    public MemorySegment getHandle() {
        return modelPtr;
    }

    @Override
    public void close() {
        release();
    }
}
