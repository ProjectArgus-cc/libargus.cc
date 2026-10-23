package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusGenerationResult;
import cc.projectargus.libargus.ArgusNativeException;
import cc.projectargus.libargus.ArgusSamplerConfig;
import java.util.Collection;
import java.util.Collections;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Shared high-level autoregressive text, multimodal generation, and embedding extraction loops
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

    /**
     * Custom prefill callback used by multimodal chunk evaluation to return the initial KV position
     * high-water mark ({@code newNPast}) after evaluating text and media chunks.
     */
    @FunctionalInterface
    public interface CustomPrefill {
        int evaluatePrefill();
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
        return generateWithResult(
            ops,
            prompt,
            maxTokens,
            config,
            Collections.emptyList(),
            shouldContinue,
            isAborted
        ).text();
    }

    /**
     * Executes high-level streaming text generation with BPE-safe stop-sequence holdback,
     * early-stopping callback predicate, and cooperative cancellation, returning rich telemetry.
     */
    public static ArgusGenerationResult generateWithResult(
        ContextOps ops,
        String prompt,
        int maxTokens,
        ArgusSamplerConfig config,
        Collection<String> stopSequences,
        Predicate<String> shouldContinue,
        BooleanSupplier isAborted
    ) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(config, "config");
        ArgusCommonValidation.checkNonNegative(maxTokens, "maxTokens");
        if (maxTokens == 0) {
            return new ArgusGenerationResult("", ArgusGenerationResult.StopReason.MAX_TOKENS, 0, 0, null);
        }

        ops.clearCacheSlot(0, 0, -1);
        ops.resetSampler(0);

        int[] promptTokens = ops.tokenize(prompt, true);
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("Prompt tokenized to 0 tokens");
        }

        int prefillStatus = ops.decodeBatch(promptTokens, promptTokens.length, 0, true);
        if (prefillStatus == STATUS_CANCELLED) {
            return new ArgusGenerationResult("", ArgusGenerationResult.StopReason.ABORTED, 0, 0, null);
        }
        if (prefillStatus != 0) {
            ArgusNativeException.checkStatus(prefillStatus, "argus_decode_tokens");
        }
        ops.primeSampler(0, promptTokens, promptTokens.length);

        return runAutoregressiveLoop(
            ops,
            promptTokens.length,
            promptTokens.length,
            maxTokens,
            config,
            stopSequences,
            shouldContinue,
            isAborted
        );
    }

    /**
     * Executes high-level streaming multimodal generation after evaluating a custom multimodal chunk prefill.
     */
    public static ArgusGenerationResult generateMultimodalWithResult(
        ContextOps ops,
        String promptForPenaltyPriming,
        CustomPrefill multimodalPrefill,
        int maxTokens,
        ArgusSamplerConfig config,
        Collection<String> stopSequences,
        Predicate<String> shouldContinue,
        BooleanSupplier isAborted
    ) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(multimodalPrefill, "multimodalPrefill");
        Objects.requireNonNull(config, "config");
        ArgusCommonValidation.checkNonNegative(maxTokens, "maxTokens");
        if (maxTokens == 0) {
            return new ArgusGenerationResult("", ArgusGenerationResult.StopReason.MAX_TOKENS, 0, 0, null);
        }
        if (isAborted != null && isAborted.getAsBoolean()) {
            return new ArgusGenerationResult("", ArgusGenerationResult.StopReason.ABORTED, 0, 0, null);
        }

        ops.clearCacheSlot(0, 0, -1);
        ops.resetSampler(0);

        int newNPast = multimodalPrefill.evaluatePrefill();
        if (newNPast < 0) {
            ArgusNativeException.checkStatus(newNPast, "argus_eval_multimodal_chunks");
        }

        if (promptForPenaltyPriming != null && !promptForPenaltyPriming.isEmpty()) {
            String strippedPrompt = promptForPenaltyPriming.replace("<__media__>", "");
            if (!strippedPrompt.isEmpty()) {
                int[] textTokens = ops.tokenize(strippedPrompt, true);
                if (textTokens.length > 0) {
                    ops.primeSampler(0, textTokens, textTokens.length);
                }
            }
        }

        return runAutoregressiveLoop(
            ops,
            newNPast,
            newNPast,
            maxTokens,
            config,
            stopSequences,
            shouldContinue,
            isAborted
        );
    }

    private static ArgusGenerationResult runAutoregressiveLoop(
        ContextOps ops,
        int initialPos,
        int promptTokenCount,
        int maxTokens,
        ArgusSamplerConfig config,
        Collection<String> stopSequences,
        Predicate<String> shouldContinue,
        BooleanSupplier isAborted
    ) {
        ArgusUtf8StreamDecoder decoder = new ArgusUtf8StreamDecoder(false);
        ArgusStopSequenceDetector stopDetector = new ArgusStopSequenceDetector(stopSequences);
        int[] singleToken = new int[1];
        int currentPos = initialPos;
        int generatedTokens = 0;
        ArgusGenerationResult.StopReason stopReason = ArgusGenerationResult.StopReason.MAX_TOKENS;

        for (int step = 0; step < maxTokens; step++) {
            if (isAborted != null && isAborted.getAsBoolean()) {
                stopReason = ArgusGenerationResult.StopReason.ABORTED;
                break;
            }
            int nextToken = ops.sampleToken(0, config);
            if (nextToken < 0) {
                ArgusNativeException.checkStatus(nextToken, "argus_sample_token_ext");
            }
            if (ops.vocabIsEog(nextToken)) {
                stopReason = ArgusGenerationResult.StopReason.EOG;
                break;
            }
            generatedTokens++;

            byte[] pieceBytes = ops.tokenToPieceBytes(nextToken);
            if (!decoder.acceptUntil(pieceBytes, 0, pieceBytes.length, piece -> stopDetector.acceptPiece(piece, shouldContinue))) {
                stopReason = stopDetector.isStoppedByStopSequence()
                    ? ArgusGenerationResult.StopReason.STOP_SEQUENCE
                    : ArgusGenerationResult.StopReason.STOPPED_BY_CALLBACK;
                break;
            }

            if (step + 1 < maxTokens) {
                singleToken[0] = nextToken;
                int stepStatus = ops.decodeBatch(singleToken, 1, currentPos, true);
                if (stepStatus == STATUS_CANCELLED) {
                    stopReason = ArgusGenerationResult.StopReason.ABORTED;
                    break;
                }
                if (stepStatus != 0) {
                    ArgusNativeException.checkStatus(stepStatus, "argus_decode_tokens");
                }
                currentPos++;
            }
        }

        boolean haltedByStopOrCallback = (stopReason == ArgusGenerationResult.StopReason.STOP_SEQUENCE
            || stopReason == ArgusGenerationResult.StopReason.STOPPED_BY_CALLBACK);
        Predicate<String> flushPredicate = (!haltedByStopOrCallback) ? shouldContinue : null;
        Consumer<String> decoderFlush = (!haltedByStopOrCallback)
            ? piece -> stopDetector.acceptPiece(piece, flushPredicate)
            : null;
        decoder.finish(decoderFlush);
        String finalText = stopDetector.finish(flushPredicate);
        if (stopDetector.isStoppedByStopSequence()) {
            stopReason = ArgusGenerationResult.StopReason.STOP_SEQUENCE;
        } else if (stopDetector.isStoppedByCallback()) {
            stopReason = ArgusGenerationResult.StopReason.STOPPED_BY_CALLBACK;
        }

        return new ArgusGenerationResult(
            finalText,
            stopReason,
            promptTokenCount,
            generatedTokens,
            stopDetector.matchedStopSequence()
        );
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
