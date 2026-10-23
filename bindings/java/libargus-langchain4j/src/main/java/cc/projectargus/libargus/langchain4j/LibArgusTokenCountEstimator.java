package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusModel;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.TokenCountEstimator;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Lock-free, thread-safe LangChain4j {@link TokenCountEstimator} backed by native {@link ArgusModel} BPE/SPM tokenization.
 */
public final class LibArgusTokenCountEstimator implements TokenCountEstimator, AutoCloseable {

    private final ArgusModel model;
    private final boolean ownsModel;
    private final LibArgusChatFormatter formatter;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusTokenCountEstimator(Builder builder) {
        ArgusBackend.init();
        if (builder.model != null) {
            this.model = builder.model;
            this.ownsModel = false;
        } else if (builder.modelPath != null) {
            this.model = ArgusModel.load(builder.modelPath, 0, false);
            this.ownsModel = true;
        } else if (builder.modelFd != null) {
            this.model = ArgusModel.loadFd(builder.modelFd, 0, false);
            this.ownsModel = true;
        } else {
            throw new IllegalArgumentException("Builder requires either model(ArgusModel), modelPath(Path), or modelFd(int)");
        }
        this.formatter = builder.chatFormatter;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Convenience factory wrapping an existing shared {@link ArgusModel} without taking ownership of its lifecycle.
     */
    public static LibArgusTokenCountEstimator fromModel(ArgusModel model) {
        return builder().model(model).build();
    }

    @Override
    public int estimateTokenCountInText(String text) {
        Objects.requireNonNull(text, "text");
        ensureOpen();
        if (text.isEmpty()) {
            return 0;
        }
        return model.tokenize(text, false).length;
    }

    @Override
    public int estimateTokenCountInMessage(ChatMessage message) {
        Objects.requireNonNull(message, "message");
        ensureOpen();
        String formatted = formatter.format(model, List.of(message), false);
        if (formatted.isEmpty()) {
            return 0;
        }
        return model.tokenize(formatted, false).length;
    }

    @Override
    public int estimateTokenCountInMessages(Iterable<ChatMessage> messages) {
        Objects.requireNonNull(messages, "messages");
        ensureOpen();
        List<ChatMessage> list = new ArrayList<>();
        for (ChatMessage msg : messages) {
            if (msg != null) {
                list.add(msg);
            }
        }
        if (list.isEmpty()) {
            return 0;
        }
        String formatted = formatter.format(model, list, false);
        if (formatted.isEmpty()) {
            return 0;
        }
        return model.tokenize(formatted, true).length;
    }

    public ArgusModel getModel() {
        return model;
    }

    public boolean isClosed() {
        return closed.get() || model.isClosed();
    }

    private void ensureOpen() {
        if (closed.get() || model.isClosed()) {
            throw new IllegalStateException("LibArgusTokenCountEstimator is already closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (ownsModel) {
            model.close();
        }
    }

    public static final class Builder {
        private ArgusModel model;
        private Path modelPath;
        private Integer modelFd;
        private LibArgusChatFormatter chatFormatter = LibArgusChatFormatter.autoDetect();

        public Builder model(ArgusModel model) {
            this.model = Objects.requireNonNull(model, "model");
            this.modelPath = null;
            this.modelFd = null;
            return this;
        }

        public Builder modelPath(Path modelPath) {
            this.modelPath = Objects.requireNonNull(modelPath, "modelPath");
            this.model = null;
            this.modelFd = null;
            return this;
        }

        public Builder modelFd(int fd) {
            if (fd < 0) {
                throw new IllegalArgumentException("Invalid file descriptor: " + fd);
            }
            this.modelFd = fd;
            this.model = null;
            this.modelPath = null;
            return this;
        }

        public Builder chatFormatter(LibArgusChatFormatter chatFormatter) {
            this.chatFormatter = Objects.requireNonNull(chatFormatter, "chatFormatter");
            return this;
        }

        public LibArgusTokenCountEstimator build() {
            return new LibArgusTokenCountEstimator(this);
        }
    }
}
