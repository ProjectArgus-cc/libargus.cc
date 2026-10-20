package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;
import cc.projectargus.libargus.internal.ArgusJniValidation;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Wraps loaded GGUF model weights in unmanaged memory via the JDK 17+ / Android-compatible JNI bridge.
 * Extends {@link ArgusJniResource} for thread-safe handle leasing and idempotent lifecycle management.
 */
public final class ArgusModel extends ArgusJniResource {

    ArgusModel(long modelHandle) {
        super(modelHandle);
    }

    @Override
    protected String resourceName() {
        return "ArgusModel";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeModelRelease(oldHandle);
    }

    /**
     * Loads GGUF weights from a local file path.
     *
     * @param modelPath path to the local GGUF file
     * @param gpuLayers number of layers to offload to VRAM
     * @param useMlock  force locking pages to RAM to prevent swapping
     * @return a loaded ArgusModel wrapper
     */
    public static ArgusModel load(Path modelPath, int gpuLayers, boolean useMlock) {
        Objects.requireNonNull(modelPath, "modelPath");
        byte[] pathUtf8 = ArgusJniValidation.toUtf8NoNull(modelPath.toAbsolutePath().toString(), "modelPath");
        long ptr = ArgusJni.nativeModelLoad(pathUtf8, gpuLayers, useMlock);
        if (ptr == 0L) {
            ArgusNativeException.throwLastError("argus_model_load for " + modelPath);
        }
        return new ArgusModel(ptr);
    }

    /**
     * Loads GGUF weights from an open POSIX file descriptor (e.g. Android AssetFileDescriptor or ParcelFileDescriptor).
     * Duplicates the file descriptor natively and rewinds the stream offset to 0 before loading; the caller retains
     * ownership of the original descriptor and may close it immediately after this call returns.
     *
     * @param fd        open POSIX file descriptor
     * @param gpuLayers number of layers to offload to VRAM
     * @param useMlock  force locking pages to RAM to prevent swapping
     * @return a loaded ArgusModel wrapper
     */
    public static ArgusModel loadFd(int fd, int gpuLayers, boolean useMlock) {
        if (fd < 0) {
            throw new IllegalArgumentException("Invalid file descriptor: " + fd);
        }
        long ptr = ArgusJni.nativeModelLoadFd(fd, gpuLayers, useMlock);
        if (ptr == 0L) {
            ArgusNativeException.throwLastError("argus_model_load_fd for fd=" + fd);
        }
        return new ArgusModel(ptr);
    }

    public int vocabBos() {
        return withHandle(ArgusJni::nativeVocabBos);
    }

    public int vocabEos() {
        return withHandle(ArgusJni::nativeVocabEos);
    }

    public int vocabEot() {
        return withHandle(ArgusJni::nativeVocabEot);
    }

    public int vocabPad() {
        return withHandle(ArgusJni::nativeVocabPad);
    }

    public int vocabNTokens() {
        return withHandle(ArgusJni::nativeVocabNTokens);
    }

    public boolean vocabIsEog(int token) {
        return withHandle(h -> ArgusJni.nativeVocabIsEog(h, token));
    }

    /**
     * Extracts model GGUF metadata string values by key name.
     *
     * @param key metadata key name
     * @return metadata value string, or null if key is missing or failed
     */
    public String getMetadataValue(String key) {
        byte[] keyUtf8 = ArgusJniValidation.toUtf8NoNull(key, "key");
        return withHandle(h -> ArgusJni.decodeUtf8(ArgusJni.nativeModelMetaValStr(h, keyUtf8)));
    }

    /**
     * Alias for {@link #getMetadataValue(String)}.
     */
    public String getMetadata(String key) {
        return getMetadataValue(key);
    }

    /**
     * Returns the number of GGUF metadata key-value pairs stored in the model.
     */
    public int getMetadataCount() {
        return withHandle(h -> Math.max(0, ArgusJni.nativeModelMetaCount(h)));
    }

    /**
     * Retrieves the GGUF metadata key string at the given index, or null if out of bounds.
     */
    public String getMetadataKeyByIndex(int index) {
        if (index < 0) {
            return null;
        }
        return withHandle(h -> ArgusJni.decodeUtf8(ArgusJni.nativeModelMetaKeyByIndex(h, index)));
    }

    /**
     * Retrieves the GGUF metadata value string at the given index, or null if out of bounds or non-string.
     */
    public String getMetadataValueByIndex(int index) {
        if (index < 0) {
            return null;
        }
        return withHandle(h -> ArgusJni.decodeUtf8(ArgusJni.nativeModelMetaValStrByIndex(h, index)));
    }

    /**
     * Traverses and retrieves all GGUF metadata key-value pairs stored in the model.
     */
    public Map<String, String> getMetadataMap() {
        return withHandle(h -> {
            Map<String, String> map = new LinkedHashMap<>();
            int count = ArgusJni.nativeModelMetaCount(h);
            if (count <= 0) {
                return map;
            }
            for (int i = 0; i < count; i++) {
                String k = ArgusJni.decodeUtf8(ArgusJni.nativeModelMetaKeyByIndex(h, i));
                String v = ArgusJni.decodeUtf8(ArgusJni.nativeModelMetaValStrByIndex(h, i));
                if (k != null && v != null) {
                    map.put(k, v);
                }
            }
            return map;
        });
    }

    public int nEmbd() {
        return withHandle(ArgusJni::nativeModelNEmbd);
    }

    public int nEmbdOut() {
        return withHandle(ArgusJni::nativeModelNEmbdOut);
    }

    public int nEmbdInp() {
        return withHandle(ArgusJni::nativeModelNEmbdInp);
    }

    public int nCtxTrain() {
        return withHandle(ArgusJni::nativeModelNCtxTrain);
    }

    public int nLayer() {
        return withHandle(ArgusJni::nativeModelNLayer);
    }

    public int nHead() {
        return withHandle(ArgusJni::nativeModelNHead);
    }

    public int nHeadKv() {
        return withHandle(ArgusJni::nativeModelNHeadKv);
    }

    public long nParams() {
        return withHandle(ArgusJni::nativeModelNParams);
    }

    public boolean hasEncoder() {
        return withHandle(ArgusJni::nativeModelHasEncoder);
    }

    public int nPosPerEmbd() {
        return withHandle(ArgusJni::nativeModelNPosPerEmbd);
    }

    public boolean isMRoPE() {
        return withHandle(ArgusJni::nativeModelIsMRoPE);
    }

    public boolean isRecurrent() {
        return withHandle(ArgusJni::nativeModelIsRecurrent);
    }

    public boolean isHybrid() {
        return withHandle(ArgusJni::nativeModelIsHybrid);
    }

    public boolean isDiffusion() {
        return withHandle(ArgusJni::nativeModelIsDiffusion);
    }

    public long modelSize() {
        return withHandle(ArgusJni::nativeModelSize);
    }

    public String desc() {
        return withHandle(h -> {
            String d = ArgusJni.decodeUtf8(ArgusJni.nativeModelDesc(h));
            return d != null ? d : "";
        });
    }

    public String getDescription() {
        return desc();
    }

    public long kvBytesPerToken(int typeK, int typeV) {
        return withHandle(h -> ArgusJni.nativeModelKvBytesPerToken(h, typeK, typeV));
    }

    public long estimateVramBytes(int contextLength, int typeK, int typeV) {
        return withHandle(h -> ArgusJni.nativeModelEstimateVramBytes(h, contextLength, typeK, typeV));
    }

    public static long quantTypeSize(int type) {
        return ArgusJni.nativeQuantTypeSize(type);
    }

    public static int quantBlockSize(int type) {
        return ArgusJni.nativeQuantBlockSize(type);
    }

    /**
     * Converts a single token ID into its raw UTF-8 byte sequence using this model's vocabulary.
     * Supports arbitrary-length token pieces via 2-tier buffer resizing.
     */
    public byte[] tokenToPieceBytes(int token) {
        return withHandle(h -> {
            byte[] bytes = ArgusJni.nativeTokenToPiece(h, token);
            return bytes != null ? bytes : new byte[0];
        });
    }

    /**
     * Converts a single token ID into its text representation (piece).
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
        return withHandle(h -> ArgusJni.nativeTokenToPieceInto(h, token, dst, offset, maxLen));
    }

    /**
     * Converts a Java string into vocabulary token IDs into a caller-supplied int array.
     *
     * @param text      Input text string to tokenize
     * @param outTokens Destination token buffer (may be null only if maxTokens == 0)
     * @param maxTokens Maximum capacity of outTokens
     * @param addBos    Whether to prepend the BOS token
     * @return Number of tokens written, or negative required token count if capacity was insufficient
     */
    public int tokenize(String text, int[] outTokens, int maxTokens, boolean addBos) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(outTokens, "outTokens");
        ArgusJniValidation.checkPositive(maxTokens, "maxTokens");
        if (maxTokens > outTokens.length) {
            throw new IllegalArgumentException("maxTokens (" + maxTokens + ") exceeds outTokens.length (" + outTokens.length + ")");
        }
        byte[] utf8 = ArgusJniValidation.toUtf8(text, "text");
        return withHandle(h -> {
            int res = ArgusJni.nativeTokenize(h, utf8, outTokens, maxTokens, addBos);
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
     * Converts a Java string into vocabulary token IDs as a primitive Java int array.
     */
    public int[] tokenize(String text, boolean addBos) {
        Objects.requireNonNull(text, "text");
        byte[] utf8 = ArgusJniValidation.toUtf8(text, "text");
        int initialCap = Math.max(64, utf8.length);
        int[] buf = new int[initialCap];
        int nTokens = withHandle(h -> {
            int res = ArgusJni.nativeTokenize(h, utf8, buf, buf.length, addBos);
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
        if (nTokens >= 0) {
            return Arrays.copyOf(buf, nTokens);
        }
        int required = -nTokens;
        if (required <= 0 || required > 10_000_000) {
            throw new IllegalArgumentException("Token count exceeds safety limit: " + required);
        }
        int[] exactBuf = new int[required];
        int retry = withHandle(h -> ArgusJni.nativeTokenize(h, utf8, exactBuf, required, addBos));
        if (retry < 0) {
            ArgusNativeException.checkStatus(retry, "argus_tokenize_n");
        }
        return retry == required ? exactBuf : Arrays.copyOf(exactBuf, retry);
    }

    // JavaBean-style getter aliases for cross-module ergonomics
    public int getVocabBos() { return vocabBos(); }
    public int getVocabEos() { return vocabEos(); }
    public int getVocabEot() { return vocabEot(); }
    public int getVocabPad() { return vocabPad(); }
    public int getVocabSize() { return vocabNTokens(); }
    public boolean isEog(int token) { return vocabIsEog(token); }
    public int getNEmbd() { return nEmbd(); }
    public int getNEmbdOut() { return nEmbdOut(); }
    public int getNEmbdInp() { return nEmbdInp(); }
    public int getNCtxTrain() { return nCtxTrain(); }
    public int getNLayer() { return nLayer(); }
    public int getNHead() { return nHead(); }
    public int getNHeadKv() { return nHeadKv(); }
    public long getNParams() { return nParams(); }
    public long getModelSize() { return modelSize(); }
}
