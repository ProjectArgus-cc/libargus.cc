package cc.projectargus.libargus;

/**
 * Configuration parameters for creating an active text generation context.
 *
 * @param draftModel       optional draft model for speculative decoding (can be null)
 * @param contextLength    total context allocation window size
 * @param cpuThreads       CPU thread ceiling for compute workloads
 * @param typeK            key cache quantization format (0 = F16, 8 = Q8_0, 2 = Q4_0, 3 = Q4_1)
 * @param typeV            value cache quantization format (0 = F16, 8 = Q8_0, 2 = Q4_0, 3 = Q4_1)
 * @param specDraftNMax    maximum tokens to evaluate speculatively per step
 * @param enableDraftMtp   enable Multi-Token Prediction (MTP) draft head
 */
public record ArgusContextConfig(
    ArgusModel draftModel,
    int contextLength,
    int cpuThreads,
    int typeK,
    int typeV,
    int specDraftNMax,
    boolean enableDraftMtp
) {
    public static final int KV_TYPE_F16 = 0;
    public static final int KV_TYPE_Q8_0 = 8;
    public static final int KV_TYPE_Q4_0 = 2;
    public static final int KV_TYPE_Q4_1 = 3;

    /**
     * Creates a default configuration optimized for general-purpose inference.
     */
    public static ArgusContextConfig createDefault(int contextLength) {
        return new ArgusContextConfig(
            null,
            contextLength,
            Math.max(1, Runtime.getRuntime().availableProcessors() / 2),
            KV_TYPE_F16,
            KV_TYPE_F16,
            0,
            false
        );
    }
}
