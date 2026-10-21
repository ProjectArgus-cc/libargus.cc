package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusCommonValidation;
import cc.projectargus.libargus.internal.ArgusJniValidation;

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
 * @param seqMax           maximum sequence slots (0 = auto-detect)
 * @param enableDraftMtp   enable Multi-Token Prediction (MTP) draft head
 * @param embeddings       enable embeddings output
 * @param kvUnified        dynamic KV cache sharing across sequences
 * @param cloneSlots       number of dedicated clone slots allocated for sequence forking
 * @param causalAttn       optional causal attention override (null uses model architecture default)
 */
public record ArgusContextConfig(
    ArgusModel draftModel,
    int contextLength,
    int cpuThreads,
    int typeK,
    int typeV,
    int specDraftNMax,
    int uBatch,
    int seqMax,
    boolean enableDraftMtp,
    boolean embeddings,
    boolean kvUnified,
    int cloneSlots,
    Boolean causalAttn
) {
    public ArgusContextConfig {
        ArgusCommonValidation.validateContextSlots(seqMax, cloneSlots);
    }

    public static final int KV_TYPE_F16 = ArgusCommonValidation.KV_TYPE_F16;
    public static final int KV_TYPE_Q8_0 = ArgusCommonValidation.KV_TYPE_Q8_0;
    public static final int KV_TYPE_Q4_0 = ArgusCommonValidation.KV_TYPE_Q4_0;
    public static final int KV_TYPE_Q4_1 = ArgusCommonValidation.KV_TYPE_Q4_1;

    public ArgusContextConfig(
        ArgusModel draftModel,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        int uBatch,
        int seqMax,
        boolean enableDraftMtp,
        boolean embeddings,
        boolean kvUnified,
        int cloneSlots
    ) {
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, uBatch, seqMax, enableDraftMtp, embeddings, kvUnified, cloneSlots, null);
    }

    public ArgusContextConfig(
        ArgusModel draftModel,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        int uBatch,
        int seqMax,
        boolean enableDraftMtp,
        boolean embeddings,
        boolean kvUnified
    ) {
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, uBatch, seqMax, enableDraftMtp, embeddings, kvUnified, 0, null);
    }

    public ArgusContextConfig(
        ArgusModel draftModel,
        int contextLength,
        int cpuThreads,
        int typeK,
        int typeV,
        int specDraftNMax,
        boolean enableDraftMtp
    ) {
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, 0, 0, enableDraftMtp, false, true);
    }

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
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, 0, 0, enableDraftMtp, embeddings, true);
    }

    public ArgusContextConfig(
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
        this(draftModel, contextLength, cpuThreads, typeK, typeV, specDraftNMax, uBatch, 0, enableDraftMtp, embeddings, true);
    }

    public static ArgusContextConfig createDefault(int contextLength) {
        return new ArgusContextConfig(
            null,
            contextLength,
            ArgusCommonValidation.defaultCpuThreads(),
            KV_TYPE_F16,
            KV_TYPE_F16,
            0,
            0,
            0,
            false,
            false,
            true,
            0
        );
    }

    public static class Builder {
        private ArgusModel draftModel = null;
        private int contextLength = 2048;
        private int cpuThreads = ArgusCommonValidation.defaultCpuThreads();
        private int typeK = KV_TYPE_F16;
        private int typeV = KV_TYPE_F16;
        private int specDraftNMax = 0;
        private int uBatch = 0;
        private int seqMax = 0;
        private int cloneSlots = 0;
        private boolean enableDraftMtp = false;
        private boolean embeddings = false;
        private boolean kvUnified = true;
        private Boolean causalAttn = null;

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

        public Builder seqMax(int seqMax) {
            this.seqMax = seqMax;
            return this;
        }

        public Builder cloneSlots(int cloneSlots) {
            ArgusJniValidation.checkNonNegative(cloneSlots, "cloneSlots");
            this.cloneSlots = cloneSlots;
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

        public Builder kvUnified(boolean kvUnified) {
            this.kvUnified = kvUnified;
            return this;
        }

        public Builder causalAttn(boolean causalAttn) {
            this.causalAttn = causalAttn;
            return this;
        }

        public ArgusContextConfig build() {
            int effectiveSeqMax = ArgusCommonValidation.computeEffectiveSeqMax(
                seqMax,
                cloneSlots,
                draftModel != null || specDraftNMax > 0 || enableDraftMtp
            );
            return new ArgusContextConfig(
                draftModel,
                contextLength,
                cpuThreads,
                typeK,
                typeV,
                specDraftNMax,
                uBatch,
                effectiveSeqMax,
                enableDraftMtp,
                embeddings,
                kvUnified,
                cloneSlots,
                causalAttn
            );
        }
    }
}
