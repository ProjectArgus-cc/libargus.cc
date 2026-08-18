package cc.projectargus.libargus;

/**
 * Immutable configuration for extended autoregressive token sampling.
 * Supported by the unmanaged zero-allocation persistent sampler pipeline.
 *
 * @param temperature        entropy control value (&lt;= 0.0f is greedy / argmax)
 * @param repeatPenalty       token repetition suppression multiplier (&lt;= 1.0f is disabled)
 * @param repeatLastN         lookback window of recent tokens to penalize (0 defaults to 64)
 * @param frequencyPenalty    frequency penalty multiplier (0.0f is disabled)
 * @param presencePenalty     presence penalty multiplier (0.0f is disabled)
 * @param topP               nucleus sampling probability ceiling (&gt;= 1.0f is disabled)
 * @param minP               minimum probability threshold relative to highest logit (&lt;= 0.0f is disabled)
 * @param topK               top-K candidate count cap (&lt;= 0 is disabled)
 * @param dryMultiplier      DRY (Don't Repeat Yourself) penalty multiplier (0.0f is disabled)
 * @param dryBase            DRY exponential base (defaults to 1.75f)
 * @param dryAllowedLength   DRY allowed common n-gram length before penalty (defaults to 2)
 * @param dryPenaltyLastN    DRY penalty lookback window (-1 or 0 matches full context)
 */
public record ArgusSamplerConfig(
    float temperature,
    float repeatPenalty,
    int repeatLastN,
    float frequencyPenalty,
    float presencePenalty,
    float topP,
    float minP,
    int topK,
    float dryMultiplier,
    float dryBase,
    int dryAllowedLength,
    int dryPenaltyLastN
) {

    /**
     * Creates a default production sampling profile (temp 0.7, topP 0.9, minP 0.05, topK 40, repeatPenalty 1.1).
     */
    public static ArgusSamplerConfig createDefault() {
        return new Builder().build();
    }

    /**
     * Creates a pure greedy / argmax sampling profile.
     */
    public static ArgusSamplerConfig greedy() {
        return new Builder().temperature(0.0f).repeatPenalty(1.0f).topP(1.0f).minP(0.0f).topK(0).build();
    }

    public static final class Builder {
        private float temperature = 0.7f;
        private float repeatPenalty = 1.1f;
        private int repeatLastN = 64;
        private float frequencyPenalty = 0.0f;
        private float presencePenalty = 0.0f;
        private float topP = 0.90f;
        private float minP = 0.05f;
        private int topK = 40;
        private float dryMultiplier = 0.0f;
        private float dryBase = 1.75f;
        private int dryAllowedLength = 2;
        private int dryPenaltyLastN = -1;

        public Builder() {}

        public Builder(ArgusSamplerConfig source) {
            if (source != null) {
                this.temperature = source.temperature();
                this.repeatPenalty = source.repeatPenalty();
                this.repeatLastN = source.repeatLastN();
                this.frequencyPenalty = source.frequencyPenalty();
                this.presencePenalty = source.presencePenalty();
                this.topP = source.topP();
                this.minP = source.minP();
                this.topK = source.topK();
                this.dryMultiplier = source.dryMultiplier();
                this.dryBase = source.dryBase();
                this.dryAllowedLength = source.dryAllowedLength();
                this.dryPenaltyLastN = source.dryPenaltyLastN();
            }
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder repeatPenalty(float repeatPenalty) {
            this.repeatPenalty = repeatPenalty;
            return this;
        }

        public Builder repeatLastN(int repeatLastN) {
            this.repeatLastN = repeatLastN;
            return this;
        }

        public Builder frequencyPenalty(float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }

        public Builder presencePenalty(float presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder minP(float minP) {
            this.minP = minP;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder dry(float multiplier, float base, int allowedLength, int penaltyLastN) {
            this.dryMultiplier = multiplier;
            this.dryBase = base;
            this.dryAllowedLength = allowedLength;
            this.dryPenaltyLastN = penaltyLastN;
            return this;
        }

        public ArgusSamplerConfig build() {
            return new ArgusSamplerConfig(
                temperature,
                repeatPenalty,
                repeatLastN,
                frequencyPenalty,
                presencePenalty,
                topP,
                minP,
                topK,
                dryMultiplier,
                dryBase,
                dryAllowedLength,
                dryPenaltyLastN
            );
        }
    }
}
