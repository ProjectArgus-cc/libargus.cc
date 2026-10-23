package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusModel;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Pluggable prompt and chat-template formatter translating a sequence of LangChain4j
 * {@link ChatMessage} instances into a model-ready prompt string (with optional {@code <__media__>}
 * markers for multimodal vision and audio evaluation).
 */
@FunctionalInterface
public interface LibArgusChatFormatter {

    /**
     * Marker token injected by {@code libmtmd} multimodal prompts for each bitmap/audio item.
     */
    String MEDIA_MARKER = "<__media__>";

    /**
     * Formats the conversation history into a prompt string for the specified model.
     *
     * @param model    active {@link ArgusModel} (may be null when formatting without model metadata)
     * @param messages non-empty list of conversation messages
     * @return formatted prompt string ready for tokenization or multimodal chunk evaluation
     */
    String format(ArgusModel model, List<ChatMessage> messages);

    /**
     * Formats the conversation history into a prompt string, optionally injecting {@code <__media__>}
     * markers for {@link ImageContent} and {@link AudioContent} items inside {@link UserMessage}s.
     */
    default String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
        return format(model, messages);
    }

    /**
     * Returns family-specific default stop sequences for the given model (such as {@code "<|im_end|>"}
     * for ChatML or {@code "<|eot_id|>"} for Llama 3).
     */
    default List<String> defaultStopSequences(ArgusModel model) {
        return List.of();
    }

    /**
     * Returns the default metadata-driven auto-detecting formatter supporting ChatML, Llama 3,
     * Gemma 2/3, Mistral, and a deterministic fallback template.
     */
    static LibArgusChatFormatter autoDetect() {
        return AutoDetectFormatter.INSTANCE;
    }

    /**
     * Returns a ChatML formatter ({@code <|im_start|>role\n...\n<|im_end|>\n}).
     */
    static LibArgusChatFormatter chatMl() {
        return TemplateFamily.CHATML;
    }

    /**
     * Returns a Llama 3 instruct formatter ({@code <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>}).
     */
    static LibArgusChatFormatter llama3() {
        return TemplateFamily.LLAMA3;
    }

    /**
     * Returns a Gemma 2 / 3 instruct formatter ({@code <start_of_turn>user/model\n...<end_of_turn>\n}).
     */
    static LibArgusChatFormatter gemma() {
        return TemplateFamily.GEMMA;
    }

    /**
     * Returns a Mistral instruct formatter ({@code [INST] ... [/INST]}).
     */
    static LibArgusChatFormatter mistral() {
        return TemplateFamily.MISTRAL;
    }

    /**
     * Returns a universal role-prefixed instruct formatter ({@code System: / User: / Assistant:}).
     */
    static LibArgusChatFormatter fallbackInstruct() {
        return TemplateFamily.FALLBACK;
    }

    /**
     * Built-in chat template families.
     */
    enum TemplateFamily implements LibArgusChatFormatter {
        CHATML {
            @Override
            public String format(ArgusModel model, List<ChatMessage> messages) {
                return format(model, messages, true);
            }

            @Override
            public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
                validateMessages(messages);
                StringBuilder sb = new StringBuilder();
                for (ChatMessage msg : messages) {
                    String role = roleOf(msg, "system", "user", "assistant", "tool");
                    sb.append("<|im_start|>").append(role).append('\n');
                    appendMessageBody(sb, msg, injectMediaMarkers);
                    sb.append("<|im_end|>\n");
                }
                sb.append("<|im_start|>assistant\n");
                return sb.toString();
            }

            @Override
            public List<String> defaultStopSequences(ArgusModel model) {
                return List.of("<|im_end|>", "<|endoftext|>");
            }
        },

        LLAMA3 {
            @Override
            public String format(ArgusModel model, List<ChatMessage> messages) {
                return format(model, messages, true);
            }

            @Override
            public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
                validateMessages(messages);
                StringBuilder sb = new StringBuilder();
                for (ChatMessage msg : messages) {
                    String role = roleOf(msg, "system", "user", "assistant", "ipython");
                    sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n");
                    appendMessageBody(sb, msg, injectMediaMarkers);
                    sb.append("<|eot_id|>");
                }
                sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
                return sb.toString();
            }

            @Override
            public List<String> defaultStopSequences(ArgusModel model) {
                return List.of("<|eot_id|>", "<|end_of_text|>");
            }
        },

        GEMMA {
            @Override
            public String format(ArgusModel model, List<ChatMessage> messages) {
                return format(model, messages, true);
            }

            @Override
            public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
                validateMessages(messages);
                StringBuilder sb = new StringBuilder();
                StringBuilder pendingSystem = new StringBuilder();
                for (ChatMessage msg : messages) {
                    if (msg instanceof SystemMessage sys) {
                        if (sys.text() != null && !sys.text().isEmpty()) {
                            if (pendingSystem.length() > 0) {
                                pendingSystem.append('\n');
                            }
                            pendingSystem.append(sys.text());
                        }
                        continue;
                    }
                    String role = (msg instanceof AiMessage) ? "model" : "user";
                    sb.append("<start_of_turn>").append(role).append('\n');
                    if (pendingSystem.length() > 0 && "user".equals(role)) {
                        sb.append(pendingSystem).append("\n\n");
                        pendingSystem.setLength(0);
                    }
                    appendMessageBody(sb, msg, injectMediaMarkers);
                    sb.append("<end_of_turn>\n");
                }
                if (pendingSystem.length() > 0) {
                    sb.append("<start_of_turn>user\n").append(pendingSystem).append("<end_of_turn>\n");
                }
                sb.append("<start_of_turn>model\n");
                return sb.toString();
            }

            @Override
            public List<String> defaultStopSequences(ArgusModel model) {
                return List.of("<end_of_turn>");
            }
        },

        MISTRAL {
            @Override
            public String format(ArgusModel model, List<ChatMessage> messages) {
                return format(model, messages, true);
            }

            @Override
            public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
                validateMessages(messages);
                StringBuilder sb = new StringBuilder();
                StringBuilder pendingSystem = new StringBuilder();
                for (ChatMessage msg : messages) {
                    if (msg instanceof SystemMessage sys) {
                        if (sys.text() != null && !sys.text().isEmpty()) {
                            if (pendingSystem.length() > 0) {
                                pendingSystem.append('\n');
                            }
                            pendingSystem.append(sys.text());
                        }
                    } else if (msg instanceof AiMessage) {
                        appendMessageBody(sb, msg, false);
                        sb.append("</s>");
                    } else {
                        sb.append("[INST] ");
                        if (pendingSystem.length() > 0) {
                            sb.append(pendingSystem).append("\n\n");
                            pendingSystem.setLength(0);
                        }
                        appendMessageBody(sb, msg, injectMediaMarkers);
                        sb.append(" [/INST]");
                    }
                }
                if (pendingSystem.length() > 0) {
                    sb.append("[INST] ").append(pendingSystem).append(" [/INST]");
                }
                return sb.toString();
            }

            @Override
            public List<String> defaultStopSequences(ArgusModel model) {
                return List.of("</s>", "[INST]");
            }
        },

        FALLBACK {
            @Override
            public String format(ArgusModel model, List<ChatMessage> messages) {
                return format(model, messages, true);
            }

            @Override
            public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
                validateMessages(messages);
                if (messages.size() == 1 && messages.get(0) instanceof UserMessage singleUser && !hasMedia(singleUser)) {
                    StringBuilder single = new StringBuilder();
                    appendMessageBody(single, singleUser, false);
                    return single.toString();
                }
                StringBuilder sb = new StringBuilder();
                for (ChatMessage msg : messages) {
                    String label = roleOf(msg, "System", "User", "Assistant", "Tool");
                    sb.append(label).append(": ");
                    appendMessageBody(sb, msg, injectMediaMarkers);
                    sb.append('\n');
                }
                sb.append("Assistant: ");
                return sb.toString();
            }

            @Override
            public List<String> defaultStopSequences(ArgusModel model) {
                return List.of("\nUser:", "\nSystem:");
            }
        };

        private static void validateMessages(List<ChatMessage> messages) {
            Objects.requireNonNull(messages, "messages");
            if (messages.isEmpty()) {
                throw new IllegalArgumentException("messages cannot be empty");
            }
        }

        private static String roleOf(ChatMessage msg, String sysRole, String userRole, String aiRole, String toolRole) {
            if (msg instanceof SystemMessage) return sysRole;
            if (msg instanceof UserMessage) return userRole;
            if (msg instanceof AiMessage) return aiRole;
            if (msg instanceof ToolExecutionResultMessage) return toolRole;
            return userRole;
        }

        private static boolean hasMedia(UserMessage userMessage) {
            for (Content c : userMessage.contents()) {
                if (c instanceof ImageContent || c instanceof AudioContent) {
                    return true;
                }
            }
            return false;
        }

        private static void appendMessageBody(StringBuilder sb, ChatMessage msg, boolean injectMediaMarkers) {
            if (msg instanceof SystemMessage sys) {
                if (sys.text() != null) {
                    sb.append(sys.text());
                }
            } else if (msg instanceof UserMessage user) {
                boolean hasEmittedText = false;
                for (Content content : user.contents()) {
                    if (content instanceof TextContent tc) {
                        if (hasEmittedText) {
                            sb.append('\n');
                        }
                        if (tc.text() != null) {
                            sb.append(tc.text());
                        }
                        hasEmittedText = true;
                    } else if ((content instanceof ImageContent || content instanceof AudioContent) && injectMediaMarkers) {
                        if (hasEmittedText && sb.length() > 0 && sb.charAt(sb.length() - 1) != '\n') {
                            sb.append('\n');
                        }
                        sb.append(MEDIA_MARKER).append('\n');
                        hasEmittedText = false;
                    }
                }
            } else if (msg instanceof AiMessage ai) {
                if (ai.text() != null && !ai.text().isEmpty()) {
                    sb.append(ai.text());
                } else if (ai.hasToolExecutionRequests()) {
                    for (ToolExecutionRequest req : ai.toolExecutionRequests()) {
                        sb.append("[ToolCall: ").append(req.name()).append('(')
                          .append(req.arguments() != null ? req.arguments() : "").append(")]");
                    }
                }
            } else if (msg instanceof ToolExecutionResultMessage tool) {
                if (tool.toolName() != null && !tool.toolName().isEmpty()) {
                    sb.append('[').append(tool.toolName()).append("] ");
                }
                if (tool.text() != null) {
                    sb.append(tool.text());
                }
            } else if (msg != null) {
                sb.append(msg);
            }
        }
    }

    /**
     * Auto-detection dispatcher inspecting GGUF {@code tokenizer.chat_template} and {@code general.architecture}.
     */
    final class AutoDetectFormatter implements LibArgusChatFormatter {
        static final AutoDetectFormatter INSTANCE = new AutoDetectFormatter();

        private AutoDetectFormatter() {}

        static TemplateFamily detectFamily(ArgusModel model) {
            if (model == null || model.isClosed()) {
                return TemplateFamily.FALLBACK;
            }
            String chatTemplate = safeMeta(model, "tokenizer.chat_template");
            if (chatTemplate != null) {
                if (chatTemplate.contains("<|im_start|>")) {
                    return TemplateFamily.CHATML;
                }
                if (chatTemplate.contains("<|start_header_id|>")) {
                    return TemplateFamily.LLAMA3;
                }
                if (chatTemplate.contains("<start_of_turn>")) {
                    return TemplateFamily.GEMMA;
                }
                if (chatTemplate.contains("[INST]")) {
                    return TemplateFamily.MISTRAL;
                }
            }
            String arch = safeMeta(model, "general.architecture");
            if (arch != null) {
                String lower = arch.toLowerCase(Locale.ROOT);
                if (lower.startsWith("qwen") || lower.startsWith("yi") || lower.contains("chatml") || lower.startsWith("deepseek")) {
                    return TemplateFamily.CHATML;
                }
                if (lower.startsWith("gemma")) {
                    return TemplateFamily.GEMMA;
                }
                if (lower.startsWith("mistral") || lower.startsWith("mixtral")) {
                    return TemplateFamily.MISTRAL;
                }
            }
            return TemplateFamily.FALLBACK;
        }

        private static String safeMeta(ArgusModel model, String key) {
            try {
                return model.getMetadataValue(key);
            } catch (RuntimeException ignored) {
                return null;
            }
        }

        @Override
        public String format(ArgusModel model, List<ChatMessage> messages) {
            return detectFamily(model).format(model, messages, true);
        }

        @Override
        public String format(ArgusModel model, List<ChatMessage> messages, boolean injectMediaMarkers) {
            return detectFamily(model).format(model, messages, injectMediaMarkers);
        }

        @Override
        public List<String> defaultStopSequences(ArgusModel model) {
            return detectFamily(model).defaultStopSequences(model);
        }
    }
}
