package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusNativeException;
import cc.projectargus.libargus.ArgusSamplerConfig;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Shared high-level autoregressive text generation and embedding extraction loops
 * shared across {@code :libargus-core} (Panama FFM) and {@code :libargus-jni} (JNI).
 */
public final class ArgusTextGenerator {

    public static final int STATUS_CANCELLED = -2;

    /**
     * Adapter interface bridging context primitives required by high-level generation and embedding helpers.
     */
    public interface ContextOps {
        boolean clearCacheSlot(int seqId, int p0, int p1);
        void resetSampler(int seqId);
        int[] tokenize(String text, boolean addBos);
        int decodeBatch(int[] tokens, int nTokens, int startPos, boolean requestLogits);
        int primeSampler(int seqId, int[] tokens, int nTokens);
        int sampleToken(int seqId, ArgusSamplerConfig config);
        boolean vocabIsEog(int token);
        byte[] tokenToPieceBytes(int token);
        int nEmbd();
        int evalEmbeddings(int[] tokens, int nTokens, int seqId, float[] outEmbeddings);
    }

    private ArgusTextGenerator() {}

    /**
     * Executes high-level streaming text generation with a {@link Consumer} callback.
     */
    public static String generate(
        ContextOps ops,
        String prompt,
        int maxTokens,
        ArgusSamplerConfig config,
        Consumer<String> onPiece,
        BooleanSupplier isAborted
    ) {
        Predicate<String> predicate = (onPiece != null) ? piece -> {
            onPiece.accept(piece);
            return true;
        } : null;
        return generateUntil(ops, prompt, maxTokens, config, predicate, isAborted);
    }

    /**
     * Executes high-level streaming text generation with an early-stopping {@link Predicate} callback.
     * If {@code shouldContinue} returns {@code false} for any emitted UTF-8 piece, generation halts immediately.
     */
    public static String generateUntil(
        ContextOps ops,
        String prompt,
        int maxTokens,
        ArgusSamplerConfig config,
        Predicate<String> shouldContinue,
        BooleanSupplier isAborted
    ) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(config, "config");
        ArgusCommonValidation.checkNonNegative(maxTokens, "maxTokens");
        if (maxTokens == 0) {
            return "";
        }

        ops.clearCacheSlot(0, 0, -1);
        ops.resetSampler(0);

        int[] promptTokens = ops.tokenize(prompt, true);
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("Prompt tokenized to 0 tokens");
        }

        int prefillStatus = ops.decodeBatch(promptTokens, promptTokens.length, 0, true);
        if (prefillStatus == STATUS_CANCELLED) {
            return "";
        }
        if (prefillStatus != 0) {
            ArgusNativeException.checkStatus(prefillStatus, "argus_decode_tokens");
        }
        ops.primeSampler(0, promptTokens, promptTokens.length);

        ArgusUtf8StreamDecoder decoder = new ArgusUtf8StreamDecoder(true);
        int[] singleToken = new int[1];
        int currentPos = promptTokens.length;
        boolean stoppedByPredicate = false;

        for (int step = 0; step < maxTokens; step++) {
            if (isAborted != null && isAborted.getAsBoolean()) {
                break;
            }
            int nextToken = ops.sampleToken(0, config);
            if (nextToken < 0) {
                ArgusNativeException.checkStatus(nextToken, "argus_sample_token_ext");
            }
            if (ops.vocabIsEog(nextToken)) {
                break;
            }

            byte[] pieceBytes = ops.tokenToPieceBytes(nextToken);
            if (!decoder.acceptUntil(pieceBytes, 0, pieceBytes.length, shouldContinue)) {
                stoppedByPredicate = true;
                break;
            }

            if (step + 1 < maxTokens) {
                singleToken[0] = nextToken;
                int stepStatus = ops.decodeBatch(singleToken, 1, currentPos, true);
                if (stepStatus == STATUS_CANCELLED) {
                    break;
                }
                if (stepStatus != 0) {
                    ArgusNativeException.checkStatus(stepStatus, "argus_decode_tokens");
                }
                currentPos++;
            }
        }

        Consumer<String> finishCallback = (!stoppedByPredicate && shouldContinue != null)
            ? shouldContinue::test
            : null;
        return decoder.finish(finishCallback);
    }

    /**
     * Extracts semantic text embeddings from a non-empty prompt into a newly allocated float array.
     */
    public static float[] extractEmbeddings(ContextOps ops, String text) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) {
            throw new IllegalArgumentException("Cannot extract embeddings for empty text string");
        }
        int[] tokens = ops.tokenize(text, true);
        if (tokens.length == 0) {
            throw new IllegalStateException("Tokenization produced 0 tokens for non-empty text");
        }
        int dim = ops.nEmbd();
        float[] result = new float[dim];
        ops.evalEmbeddings(tokens, tokens.length, 0, result);
        return result;
    }
}
