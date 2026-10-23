package cc.projectargus.libargus;

import java.util.Objects;

/**
 * Immutable telemetry and output record returned by high-level text and multimodal generation loops.
 *
 * @param text                complete generated text (excluding any matched trailing stop sequence)
 * @param stopReason          deterministic reason why autoregressive generation terminated
 * @param promptTokens        number of prompt tokens (or multimodal positions) evaluated during prefill
 * @param generatedTokens     number of output tokens sampled during generation
 * @param matchedStopSequence the exact stop sequence that halted generation when {@code stopReason == STOP_SEQUENCE}, or {@code null}
 */
public record ArgusGenerationResult(
    String text,
    StopReason stopReason,
    int promptTokens,
    int generatedTokens,
    String matchedStopSequence
) {
    /**
     * Enumeration of terminal states for an autoregressive generation loop.
     */
    public enum StopReason {
        /** Model sampled an End-of-Generation (EOG / EOS / EOT) vocabulary token. */
        EOG,
        /** Generation reached the caller-specified {@code maxTokens} ceiling. */
        MAX_TOKENS,
        /** Generation matched a configured stop sequence (which was stripped from {@link #text()}). */
        STOP_SEQUENCE,
        /** Streaming callback predicate returned {@code false} to halt generation early. */
        STOPPED_BY_CALLBACK,
        /** Cooperative cancellation flag ({@code ArgusAbortFlag}) was triggered before or during decoding. */
        ABORTED
    }

    public ArgusGenerationResult {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(stopReason, "stopReason");
        if (promptTokens < 0) {
            throw new IllegalArgumentException("promptTokens must be >= 0 (got " + promptTokens + ")");
        }
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must be >= 0 (got " + generatedTokens + ")");
        }
    }
}
