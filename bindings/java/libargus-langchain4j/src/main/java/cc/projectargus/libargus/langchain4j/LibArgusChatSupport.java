package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusGenerationResult;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.ArgusPerfTimings;
import cc.projectargus.libargus.ArgusSamplerConfig;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Shared parameter resolution, stop-sequence merging, and telemetry mapping utilities
 * for {@link LibArgusChatModel} and {@link LibArgusStreamingChatModel}.
 */
final class LibArgusChatSupport {

    private LibArgusChatSupport() {}

    static ArgusSamplerConfig resolveSamplerConfig(ArgusSamplerConfig base, ChatRequestParameters params) {
        if (params == null) {
            return base;
        }
        ArgusSamplerConfig.Builder builder = new ArgusSamplerConfig.Builder(base);
        if (params.temperature() != null) {
            builder.temperature(params.temperature().floatValue());
        }
        if (params.topP() != null) {
            builder.topP(params.topP().floatValue());
        }
        if (params.topK() != null) {
            builder.topK(params.topK());
        }
        if (params.frequencyPenalty() != null) {
            builder.frequencyPenalty(params.frequencyPenalty().floatValue());
        }
        if (params.presencePenalty() != null) {
            builder.presencePenalty(params.presencePenalty().floatValue());
        }
        return builder.build();
    }

    static int resolveMaxTokens(int defaultMaxTokens, ChatRequestParameters params) {
        if (params != null && params.maxOutputTokens() != null) {
            int reqMax = params.maxOutputTokens();
            if (reqMax < 0) {
                throw new IllegalArgumentException("maxOutputTokens must be >= 0 (got " + reqMax + ")");
            }
            return reqMax;
        }
        return defaultMaxTokens;
    }

    static List<String> resolveStopSequences(
        List<String> builderStops,
        LibArgusChatFormatter formatter,
        ArgusModel model,
        ChatRequestParameters params
    ) {
        Set<String> merged = new LinkedHashSet<>();
        if (formatter != null) {
            List<String> fmtStops = formatter.defaultStopSequences(model);
            if (fmtStops != null) {
                for (String s : fmtStops) {
                    if (s != null && !s.isEmpty()) {
                        merged.add(s);
                    }
                }
            }
        }
        if (builderStops != null) {
            for (String s : builderStops) {
                if (s != null && !s.isEmpty()) {
                    merged.add(s);
                }
            }
        }
        if (params != null && params.stopSequences() != null) {
            for (String s : params.stopSequences()) {
                if (s != null && !s.isEmpty()) {
                    merged.add(s);
                }
            }
        }
        return new ArrayList<>(merged);
    }

    static FinishReason toFinishReason(ArgusGenerationResult.StopReason stopReason) {
        if (stopReason == null) {
            return FinishReason.OTHER;
        }
        return switch (stopReason) {
            case EOG, STOP_SEQUENCE -> FinishReason.STOP;
            case MAX_TOKENS -> FinishReason.LENGTH;
            case ABORTED, STOPPED_BY_CALLBACK -> FinishReason.OTHER;
        };
    }

    static ChatResponse buildChatResponse(ArgusContext context, ArgusGenerationResult result) {
        ArgusPerfTimings perf = context.getPerfTimings();
        int inputTokens = result.promptTokens() > 0 ? result.promptTokens() : Math.max(0, perf.nPromptEval());
        int outputTokens = result.generatedTokens();
        TokenUsage usage = new TokenUsage(inputTokens, outputTokens);
        return ChatResponse.builder()
            .aiMessage(AiMessage.from(result.text()))
            .tokenUsage(usage)
            .finishReason(toFinishReason(result.stopReason()))
            .build();
    }
}
