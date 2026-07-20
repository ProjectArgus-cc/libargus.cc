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
 * @param uBatch           physical micro-batch size (0 = default/auto)
 * @param enableDraftMtp   enable Multi-Token Prediction (MTP) draft head
 * @param embeddings       enable embeddings output
 */
public record ArgusContextConfig(
    ArgusModel draftModel,
    int contextLength,
    int cpuThreads,
    int typeK,
    int typeV,
    int specDraftNMax,
    int uBatch,
    boolean enableDraftMtp,
    boolean embeddings
) {
    public static final int KV_TYPE_F16 = 0;
    public static final int KV_TYPE_Q8_0 = 8;
    public static final int KV_TYPE_Q4_0 = 2;
    public static final int KV_TYPE_Q4_1 = 3;

    /**
     * Backwards-compatible constructor without uBatch or embeddings parameters.
     */
    public ArgusContextConfig(
        ArgusModel draftModel,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        boolean enableDraftMtp
    ) {
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, 0, enableDraftMtp, false);
    }

    /**
     * Backwards-compatible constructor without uBatch parameter.
     */
    public ArgusContextConfig(
        ArgusModel draftModel,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        boolean enableDraftMtp,
        boolean embeddings
    ) {
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, 0, enableDraftMtp, embeddings);
    }

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
            0,
            false,
            false
        );
    }

    /**
     * Fluent, zero-dependency builder for creating ArgusContextConfig instances.
     */
    public static class Builder {
        private ArgusModel draftModel = null;
        private int contextLength = 2048;
        private int cpuThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        private int typeK = KV_TYPE_F16;
        private int typeV = KV_TYPE_F16;
        private int specDraftNMax = 0;
        private int uBatch = 0;
        private boolean enableDraftMtp = false;
        private boolean embeddings = false;

        public Builder() {}

        public Builder(int contextLength) {
            this.contextLength = contextLength;
        }

        public Builder draftModel(ArgusModel draftModel) {
            this.draftModel = draftModel;
            return this;
        }

        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder cpuThreads(int cpuThreads) {
            this.cpuThreads = cpuThreads;
            return this;
        }

        public Builder typeK(int typeK) {
            this.typeK = typeK;
            return this;
        }

        public Builder typeV(int typeV) {
            this.typeV = typeV;
            return this;
        }

        public Builder specDraftNMax(int specDraftNMax) {
            this.specDraftNMax = specDraftNMax;
            return this;
        }

        public Builder uBatch(int uBatch) {
            this.uBatch = uBatch;
            return this;
        }

        public Builder enableDraftMtp(boolean enableDraftMtp) {
            this.enableDraftMtp = enableDraftMtp;
            return this;
        }

        public Builder embeddings(boolean embeddings) {
            this.embeddings = embeddings;
            return this;
        }

        public ArgusContextConfig build() {
            return new ArgusContextConfig(
                draftModel,
                contextLength,
                cpuThreads,
                typeK,
                typeV,
                specDraftNMax,
                uBatch,
                enableDraftMtp,
                embeddings
            );
        }
    }
}
