package cc.projectargus.libargus;

/**
 * Immutable record holding performance execution telemetry for an active context session.
 *
 * @param startTimeMs Absolute start timestamp in milliseconds
 * @param loadTimeMs Model loading duration in milliseconds
 * @param promptEvalTimeMs Prompt prefill processing duration in milliseconds
 * @param evalTimeMs Autoregressive token evaluation duration in milliseconds
 * @param nPromptEval Evaluated prompt token count
 * @param nEval Evaluated generation token count
 * @param nReused Reused graph compute iterations count
 */
public record ArgusPerfTimings(
    double startTimeMs,
    double loadTimeMs,
    double promptEvalTimeMs,
    double evalTimeMs,
    int nPromptEval,
    int nEval,
    int nReused
) {
    /**
     * Calculates the prompt prefill processing speed in tokens per second.
     * @return Tokens per second, or 0.0 if not evaluated.
     */
    public double promptTokensPerSecond() {
        return (promptEvalTimeMs > 0.0 && nPromptEval > 0)
            ? (nPromptEval / (promptEvalTimeMs / 1000.0))
            : 0.0;
    }

    /**
     * Calculates the autoregressive generation speed in tokens per second.
     * @return Tokens per second, or 0.0 if not evaluated.
     */
    public double evalTokensPerSecond() {
        return (evalTimeMs > 0.0 && nEval > 0)
            ? (nEval / (evalTimeMs / 1000.0))
            : 0.0;
    }
}
