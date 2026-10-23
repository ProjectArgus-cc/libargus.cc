package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusAbortFlag;
import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusBitmap;
import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.ArgusGenerationResult;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.ArgusMultimodalContext;
import cc.projectargus.libargus.ArgusSamplerConfig;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * In-process, GPU-accelerated LangChain4j {@link ChatModel} backed by {@code libargus}.
 *
 * <p>Binary-compatible with both {@code :libargus-jni} (JDK 17+ / Android) and
 * {@code :libargus-core} (JDK 22+ Project Panama FFM).
 */
public final class LibArgusChatModel implements ChatModel, AutoCloseable {

    private final ArgusModel model;
    private final boolean ownsModel;
    private final ArgusContext context;
    private final ArgusMultimodalContext multimodalContext;
    private final ArgusSamplerConfig defaultSampler;
    private final int defaultMaxTokens;
    private final List<String> defaultStopSequences;
    private final LibArgusChatFormatter formatter;
    private final Path allowedMediaRoot;
    private final boolean allowLocalFileMedia;
    private final long maxMediaBytes;
    private final ReentrantLock executionLock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusChatModel(Builder builder) {
        ArgusBackend.init();

        final ArgusModel resolvedModel;
        final boolean modelOwned;
        if (builder.model != null) {
            resolvedModel = builder.model;
            modelOwned = false;
        } else if (builder.modelPath != null) {
            resolvedModel = ArgusModel.load(builder.modelPath, builder.gpuLayers, builder.useMlock);
            modelOwned = true;
        } else if (builder.modelFd != null) {
            resolvedModel = ArgusModel.loadFd(builder.modelFd, builder.gpuLayers, builder.useMlock);
            modelOwned = true;
        } else {
            throw new IllegalArgumentException("Builder requires either model(ArgusModel), modelPath(Path), or modelFd(int)");
        }

        ArgusContext createdContext = null;
        ArgusMultimodalContext createdMctx = null;
        try {
            ArgusContextConfig ctxConfig = builder.buildContextConfig();
            createdContext = ArgusContext.init(resolvedModel, ctxConfig);
            if (builder.mmprojPath != null) {
                createdMctx = ArgusMultimodalContext.init(
                    resolvedModel,
                    builder.mmprojPath,
                    ctxConfig.cpuThreads(),
                    builder.multimodalUseGpu
                );
            }
        } catch (Throwable t) {
            if (createdMctx != null) {
                try { createdMctx.close(); } catch (Throwable ignored) {}
            }
            if (createdContext != null) {
                try { createdContext.close(); } catch (Throwable ignored) {}
            }
            if (modelOwned) {
                try { resolvedModel.close(); } catch (Throwable ignored) {}
            }
            throw t;
        }

        this.model = resolvedModel;
        this.ownsModel = modelOwned;
        this.context = createdContext;
        this.multimodalContext = createdMctx;
        this.defaultSampler = builder.samplerConfig;
        this.defaultMaxTokens = builder.maxOutputTokens;
        this.defaultStopSequences = List.copyOf(builder.stopSequences);
        this.formatter = builder.chatFormatter;
        this.allowedMediaRoot = builder.allowedMediaRoot;
        this.allowLocalFileMedia = builder.allowLocalFileMedia;
        this.maxMediaBytes = builder.maxMediaBytes;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public ChatResponse doChat(ChatRequest chatRequest) {
        return doChat(chatRequest, null);
    }

    /**
     * Executes a synchronous chat request with optional cooperative cancellation via {@link ArgusAbortFlag}.
     */
    public ChatResponse doChat(ChatRequest chatRequest, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(chatRequest, "chatRequest");
        List<ChatMessage> messages = chatRequest.messages();
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("chatRequest.messages() cannot be null or empty");
        }
        ensureOpen();

        ChatRequestParameters params = chatRequest.parameters();
        ArgusSamplerConfig sampler = LibArgusChatSupport.resolveSamplerConfig(defaultSampler, params);
        int maxTokens = LibArgusChatSupport.resolveMaxTokens(defaultMaxTokens, params);
        List<String> stopSequences = LibArgusChatSupport.resolveStopSequences(
            defaultStopSequences, formatter, model, params
        );

        executionLock.lock();
        try {
            ensureOpen();
            context.resetPerfTimings();

            boolean hasMedia = LibArgusMediaSupport.hasMediaContent(messages);
            if (hasMedia && multimodalContext == null) {
                throw new IllegalStateException(
                    "ChatRequest contains multimodal ImageContent/AudioContent, but no mmprojPath was configured on LibArgusChatModel"
                );
            }

            final ArgusGenerationResult result;
            if (hasMedia) {
                String prompt = formatter.format(model, messages, true);
                List<ArgusBitmap> bitmaps = LibArgusMediaSupport.extractBitmaps(
                    multimodalContext,
                    messages,
                    allowedMediaRoot,
                    allowLocalFileMedia,
                    maxMediaBytes
                );
                try {
                    result = context.generateMultimodalWithResult(
                        multimodalContext,
                        prompt,
                        bitmaps,
                        0,
                        maxTokens,
                        sampler,
                        stopSequences,
                        (Consumer<String>) null,
                        abortFlag
                    );
                } finally {
                    LibArgusMediaSupport.closeAllQuietly(bitmaps);
                }
            } else {
                String prompt = formatter.format(model, messages, false);
                result = context.generateWithResult(
                    prompt,
                    maxTokens,
                    sampler,
                    stopSequences,
                    (Consumer<String>) null,
                    abortFlag
                );
            }

            return LibArgusChatSupport.buildChatResponse(context, result);
        } finally {
            executionLock.unlock();
        }
    }

    public ArgusModel getModel() {
        return model;
    }

    public ArgusContext getContext() {
        return context;
    }

    public boolean isClosed() {
        return closed.get() || context.isClosed();
    }

    private void ensureOpen() {
        if (closed.get() || context.isClosed()) {
            throw new IllegalStateException("LibArgusChatModel is already closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        executionLock.lock();
        try {
            if (multimodalContext != null) {
                try {
                    multimodalContext.close();
                } catch (Throwable ignored) {
                }
            }
            try {
                context.close();
            } finally {
                if (ownsModel) {
                    model.close();
                }
            }
        } finally {
            executionLock.unlock();
        }
    }

    public static final class Builder {
        private ArgusModel model;
        private Path modelPath;
        private Integer modelFd;
        private int gpuLayers = 99;
        private boolean useMlock = false;

        private ArgusContextConfig contextConfig;
        private int contextLength = 2048;
        private int cpuThreads = ArgusCommonValidation.defaultCpuThreads();
        private int typeK = ArgusContextConfig.KV_TYPE_F16;
        private int typeV = ArgusContextConfig.KV_TYPE_F16;

        private ArgusSamplerConfig samplerConfig = ArgusSamplerConfig.createDefault();
        private int maxOutputTokens = 512;
        private List<String> stopSequences = Collections.emptyList();
        private LibArgusChatFormatter chatFormatter = LibArgusChatFormatter.autoDetect();

        private Path mmprojPath;
        private boolean multimodalUseGpu = true;
        private Path allowedMediaRoot;
        private boolean allowLocalFileMedia = false;
        private long maxMediaBytes = LibArgusMediaSupport.DEFAULT_MAX_MEDIA_BYTES;

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

        public Builder gpuLayers(int gpuLayers) {
            this.gpuLayers = gpuLayers;
            return this;
        }

        public Builder useMlock(boolean useMlock) {
            this.useMlock = useMlock;
            return this;
        }

        public Builder contextConfig(ArgusContextConfig contextConfig) {
            this.contextConfig = Objects.requireNonNull(contextConfig, "contextConfig");
            return this;
        }

        public Builder contextLength(int contextLength) {
            ArgusCommonValidation.checkPositive(contextLength, "contextLength");
            this.contextLength = contextLength;
            return this;
        }

        public Builder cpuThreads(int cpuThreads) {
            ArgusCommonValidation.checkPositive(cpuThreads, "cpuThreads");
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

        public Builder samplerConfig(ArgusSamplerConfig samplerConfig) {
            this.samplerConfig = Objects.requireNonNull(samplerConfig, "samplerConfig");
            return this;
        }

        public Builder temperature(float temperature) {
            this.samplerConfig = new ArgusSamplerConfig.Builder(this.samplerConfig).temperature(temperature).build();
            return this;
        }

        public Builder topP(float topP) {
            this.samplerConfig = new ArgusSamplerConfig.Builder(this.samplerConfig).topP(topP).build();
            return this;
        }

        public Builder topK(int topK) {
            this.samplerConfig = new ArgusSamplerConfig.Builder(this.samplerConfig).topK(topK).build();
            return this;
        }

        public Builder seed(int seed) {
            this.samplerConfig = new ArgusSamplerConfig.Builder(this.samplerConfig).seed(seed).build();
            return this;
        }

        public Builder maxOutputTokens(int maxOutputTokens) {
            ArgusCommonValidation.checkNonNegative(maxOutputTokens, "maxOutputTokens");
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder stopSequences(Collection<String> stopSequences) {
            Objects.requireNonNull(stopSequences, "stopSequences");
            List<String> copy = new ArrayList<>(stopSequences.size());
            for (String s : stopSequences) {
                if (s != null && !s.isEmpty()) {
                    copy.add(s);
                }
            }
            this.stopSequences = Collections.unmodifiableList(copy);
            return this;
        }

        public Builder chatFormatter(LibArgusChatFormatter chatFormatter) {
            this.chatFormatter = Objects.requireNonNull(chatFormatter, "chatFormatter");
            return this;
        }

        public Builder mmprojPath(Path mmprojPath) {
            this.mmprojPath = mmprojPath;
            return this;
        }

        public Builder multimodalUseGpu(boolean multimodalUseGpu) {
            this.multimodalUseGpu = multimodalUseGpu;
            return this;
        }

        public Builder allowedMediaRoot(Path allowedMediaRoot) {
            this.allowedMediaRoot = allowedMediaRoot;
            return this;
        }

        public Builder allowLocalFileMedia(boolean allowLocalFileMedia) {
            this.allowLocalFileMedia = allowLocalFileMedia;
            return this;
        }

        public Builder maxMediaBytes(long maxMediaBytes) {
            if (maxMediaBytes <= 0) {
                throw new IllegalArgumentException("maxMediaBytes must be > 0 (got " + maxMediaBytes + ")");
            }
            this.maxMediaBytes = maxMediaBytes;
            return this;
        }

        private ArgusContextConfig buildContextConfig() {
            if (contextConfig != null) {
                return contextConfig;
            }
            return new ArgusContextConfig.Builder(contextLength)
                .cpuThreads(cpuThreads)
                .typeK(typeK)
                .typeV(typeV)
                .build();
        }

        public LibArgusChatModel build() {
            return new LibArgusChatModel(this);
        }
    }
}
