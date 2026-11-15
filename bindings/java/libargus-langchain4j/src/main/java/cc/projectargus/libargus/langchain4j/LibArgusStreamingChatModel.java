package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusAbortFlag;
import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.ArgusGenerationResult;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.ArgusMultimodalContext;
import cc.projectargus.libargus.ArgusSamplerConfig;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatModelStreamingEvent;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * In-process, GPU-accelerated LangChain4j {@link StreamingChatModel} backed by {@code libargus}.
 *
 * <p>Streams RFC 3629-compliant UTF-8 character sequences token-by-token while holding back
 * partial stop-sequence and {@code <think>} prefixes, wiring {@link StreamingHandle#cancel()}
 * and reactive {@link Flow.Subscription#cancel()} directly into native {@link ArgusAbortFlag}
 * cooperative cancellation.
 */
public final class LibArgusStreamingChatModel implements StreamingChatModel, AutoCloseable {

    private final ArgusModel model;
    private final boolean ownsModel;
    private final ArgusContext context;
    private final ArgusMultimodalContext multimodalContext;
    private final ArgusSamplerConfig defaultSampler;
    private final int defaultMaxTokens;
    private final List<String> defaultStopSequences;
    private final LibArgusChatFormatter formatter;
    private final boolean returnThinking;
    private final String modelName;
    private final ChatRequestParameters defaultRequestParameters;
    private final List<ChatModelListener> listeners;
    private final Set<Capability> supportedCapabilities;
    private final Executor asyncExecutor;
    private final Path allowedMediaRoot;
    private final boolean allowLocalFileMedia;
    private final long maxMediaBytes;
    private final float videoFps;
    private final long videoTimestampIntervalMs;
    private final int maxVideoFrames;
    private final ReentrantLock executionLock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusStreamingChatModel(Builder builder) {
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
        this.returnThinking = builder.returnThinking;
        this.modelName = builder.modelName;
        this.defaultRequestParameters = builder.buildDefaultRequestParameters();
        this.listeners = List.copyOf(builder.listeners);
        this.supportedCapabilities = Collections.unmodifiableSet(new LinkedHashSet<>(builder.supportedCapabilities));
        this.asyncExecutor = builder.asyncExecutor;
        this.allowedMediaRoot = builder.allowedMediaRoot;
        this.allowLocalFileMedia = builder.allowLocalFileMedia;
        this.maxMediaBytes = builder.maxMediaBytes;
        this.videoFps = builder.videoFps;
        this.videoTimestampIntervalMs = builder.videoTimestampIntervalMs;
        this.maxVideoFrames = builder.maxVideoFrames;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<ChatModelListener> listeners() {
        return listeners;
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return defaultRequestParameters;
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return supportedCapabilities;
    }

    @Override
    public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
        doChat(chatRequest, handler, null);
    }

    @Override
    public Flow.Publisher<ChatModelStreamingEvent> doChat(ChatRequest chatRequest) {
        if (asyncExecutor == null) {
            return StreamingChatModel.super.doChat(chatRequest);
        }
        Objects.requireNonNull(chatRequest, "chatRequest");
        ensureOpen();
        return subscriber -> {
            Objects.requireNonNull(subscriber, "subscriber");
            ArgusStreamingSubscription subscription = new ArgusStreamingSubscription(chatRequest, subscriber);
            subscriber.onSubscribe(subscription);
        };
    }

    /**
     * Executes a streaming chat request with an optional per-request {@link ArgusAbortFlag} for cooperative cancellation.
     */
    public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler, ArgusAbortFlag abortFlag) {
        Objects.requireNonNull(chatRequest, "chatRequest");
        Objects.requireNonNull(handler, "handler");
        List<ChatMessage> messages = chatRequest.messages();
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("chatRequest.messages() cannot be null or empty");
        }
        ensureOpen();

        ChatRequestParameters params = LibArgusChatSupport.resolveEffectiveParameters(
            defaultRequestParameters,
            chatRequest.parameters()
        );
        ArgusSamplerConfig sampler = LibArgusChatSupport.resolveSamplerConfig(defaultSampler, params);
        int maxTokens = LibArgusChatSupport.resolveMaxTokens(defaultMaxTokens, params);
        List<String> stopSequences = LibArgusChatSupport.resolveStopSequences(
            defaultStopSequences, formatter, model, params
        );

        final ArgusAbortFlag ownedAbortFlag = (abortFlag == null) ? new ArgusAbortFlag() : null;
        final ArgusAbortFlag effectiveAbortFlag = (abortFlag != null) ? abortFlag : ownedAbortFlag;
        final AtomicBoolean handleCancelled = new AtomicBoolean(false);
        final StreamingHandle streamingHandle = new StreamingHandle() {
            @Override
            public void cancel() {
                handleCancelled.set(true);
                effectiveAbortFlag.abort();
            }

            @Override
            public boolean isCancelled() {
                return handleCancelled.get() || effectiveAbortFlag.isAborted();
            }
        };
        final PartialResponseContext partialResponseCtx = new PartialResponseContext(streamingHandle);
        final PartialThinkingContext partialThinkingCtx = new PartialThinkingContext(streamingHandle);

        executionLock.lock();
        try {
            ensureOpen();
            context.resetPerfTimings();

            boolean hasMedia = LibArgusMediaSupport.hasMediaContent(messages);
            if (hasMedia && multimodalContext == null) {
                throw new IllegalStateException(
                    "ChatRequest contains multimodal ImageContent/AudioContent/VideoContent, but no mmprojPath was configured on LibArgusStreamingChatModel"
                );
            }

            final String effectivePrompt;
            final LibArgusThinkingParser thinkingParser;
            final Consumer<String> onPiece;
            final ArgusGenerationResult result;

            if (hasMedia) {
                String formattedPrompt = formatter.format(model, messages, true);
                LibArgusMediaSupport.PreparedMultimodalInput prepared = LibArgusMediaSupport.prepareMultimodalInput(
                    multimodalContext,
                    formattedPrompt,
                    messages,
                    allowedMediaRoot,
                    allowLocalFileMedia,
                    maxMediaBytes,
                    videoFps,
                    videoTimestampIntervalMs,
                    maxVideoFrames
                );
                effectivePrompt = prepared.prompt();
                thinkingParser = createStreamingThinkingParser(
                    effectivePrompt, handler, streamingHandle, partialResponseCtx, partialThinkingCtx
                );
                onPiece = createPieceConsumer(thinkingParser, handler, streamingHandle, partialResponseCtx);
                try {
                    result = context.generateMultimodalWithResult(
                        multimodalContext,
                        effectivePrompt,
                        prepared.bitmaps(),
                        0,
                        maxTokens,
                        sampler,
                        stopSequences,
                        onPiece,
                        effectiveAbortFlag
                    );
                } finally {
                    LibArgusMediaSupport.closeAllQuietly(prepared.bitmaps());
                }
            } else {
                effectivePrompt = formatter.format(model, messages, false);
                thinkingParser = createStreamingThinkingParser(
                    effectivePrompt, handler, streamingHandle, partialResponseCtx, partialThinkingCtx
                );
                onPiece = createPieceConsumer(thinkingParser, handler, streamingHandle, partialResponseCtx);
                result = context.generateWithResult(
                    effectivePrompt,
                    maxTokens,
                    sampler,
                    stopSequences,
                    onPiece,
                    effectiveAbortFlag
                );
            }

            LibArgusThinkingParser.ParsedOutput parsedThinking =
                thinkingParser != null ? thinkingParser.finish() : null;
            String effectiveModelName = LibArgusChatSupport.resolveModelName(modelName, params, model);
            ChatResponse response = LibArgusChatSupport.buildChatResponse(
                context, result, effectiveModelName, parsedThinking
            );
            handler.onCompleteResponse(response);
        } catch (Throwable t) {
            try {
                handler.onError(t);
            } catch (Throwable callbackErr) {
                t.addSuppressed(callbackErr);
                if (t instanceof RuntimeException re) {
                    throw re;
                }
                throw new RuntimeException(t);
            }
        } finally {
            try {
                if (ownedAbortFlag != null) {
                    ownedAbortFlag.close();
                }
            } catch (Throwable ignored) {
            } finally {
                executionLock.unlock();
            }
        }
    }

    private LibArgusThinkingParser createStreamingThinkingParser(
        String effectivePrompt,
        StreamingChatResponseHandler handler,
        StreamingHandle handle,
        PartialResponseContext partialResponseCtx,
        PartialThinkingContext partialThinkingCtx
    ) {
        if (!returnThinking) {
            return null;
        }
        boolean promptEndsWithOpenThink = LibArgusThinkingParser.promptEndsWithOpenThinkTag(effectivePrompt);
        return new LibArgusThinkingParser(promptEndsWithOpenThink, (chunk, isThinking) -> {
            if (handle.isCancelled() || chunk.isEmpty()) {
                return;
            }
            if (isThinking) {
                handler.onPartialThinking(new PartialThinking(chunk), partialThinkingCtx);
            } else {
                handler.onPartialResponse(new PartialResponse(chunk), partialResponseCtx);
            }
        });
    }

    private static Consumer<String> createPieceConsumer(
        LibArgusThinkingParser thinkingParser,
        StreamingChatResponseHandler handler,
        StreamingHandle handle,
        PartialResponseContext partialResponseCtx
    ) {
        if (thinkingParser != null) {
            return piece -> {
                if (!handle.isCancelled() && piece != null && !piece.isEmpty()) {
                    thinkingParser.accept(piece);
                }
            };
        }
        return piece -> {
            if (!handle.isCancelled() && piece != null && !piece.isEmpty()) {
                handler.onPartialResponse(new PartialResponse(piece), partialResponseCtx);
            }
        };
    }

    private final class ArgusStreamingSubscription implements Flow.Subscription {
        private final ChatRequest request;
        private final Flow.Subscriber<? super ChatModelStreamingEvent> subscriber;
        private final ConcurrentLinkedQueue<ChatModelStreamingEvent> queue = new ConcurrentLinkedQueue<>();
        private final AtomicLong requested = new AtomicLong(0L);
        private final AtomicInteger wip = new AtomicInteger(0);
        private final AtomicBoolean workerStarted = new AtomicBoolean(false);
        private final AtomicBoolean cancelled = new AtomicBoolean(false);
        private final AtomicBoolean producerDone = new AtomicBoolean(false);
        private final AtomicReference<Throwable> terminalError = new AtomicReference<>(null);
        private volatile ArgusAbortFlag workerAbortFlag;

        ArgusStreamingSubscription(
            ChatRequest request,
            Flow.Subscriber<? super ChatModelStreamingEvent> subscriber
        ) {
            this.request = request;
            this.subscriber = subscriber;
        }

        @Override
        public void request(long n) {
            if (cancelled.get()) {
                return;
            }
            if (n <= 0L) {
                cancelled.set(true);
                ArgusAbortFlag flag = workerAbortFlag;
                if (flag != null) {
                    try { flag.abort(); } catch (Throwable ignored) {}
                }
                subscriber.onError(new IllegalArgumentException(
                    "Flow.Subscription.request(n) requires n > 0 (Reactive Streams Rule 3.9), got " + n
                ));
                return;
            }
            addDemandCapMax(requested, n);
            if (workerStarted.compareAndSet(false, true)) {
                startWorker();
            }
            drain();
        }

        @Override
        public void cancel() {
            if (cancelled.compareAndSet(false, true)) {
                ArgusAbortFlag flag = workerAbortFlag;
                if (flag != null) {
                    try {
                        flag.abort();
                    } catch (Throwable ignored) {
                    }
                }
                queue.clear();
            }
        }

        private void startWorker() {
            ArgusAbortFlag flag = new ArgusAbortFlag();
            this.workerAbortFlag = flag;
            if (cancelled.get()) {
                flag.abort();
            }
            try {
                asyncExecutor.execute(() -> {
                    try {
                        if (cancelled.get()) {
                            return;
                        }
                        doChat(request, new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialThinking(PartialThinking partialThinking, PartialThinkingContext ctx) {
                                if (!cancelled.get()) {
                                    queue.offer(partialThinking);
                                    drain();
                                }
                            }

                            @Override
                            public void onPartialResponse(PartialResponse partialResponse, PartialResponseContext ctx) {
                                if (!cancelled.get()) {
                                    queue.offer(partialResponse);
                                    drain();
                                }
                            }

                            @Override
                            public void onCompleteResponse(ChatResponse completeResponse) {
                                if (!cancelled.get()) {
                                    queue.offer(new CompleteResponse(completeResponse));
                                    producerDone.set(true);
                                    drain();
                                }
                            }

                            @Override
                            public void onError(Throwable error) {
                                if (!cancelled.get()) {
                                    terminalError.compareAndSet(null, error);
                                    producerDone.set(true);
                                    drain();
                                }
                            }
                        }, flag);
                    } catch (Throwable unexpected) {
                        if (!cancelled.get()) {
                            terminalError.compareAndSet(null, unexpected);
                            producerDone.set(true);
                            drain();
                        }
                    } finally {
                        try {
                            flag.close();
                        } catch (Throwable ignored) {
                        }
                    }
                });
            } catch (Throwable submitError) {
                try {
                    flag.close();
                } catch (Throwable ignored) {
                }
                if (!cancelled.get()) {
                    terminalError.compareAndSet(null, submitError);
                    producerDone.set(true);
                    drain();
                }
            }
        }

        private void drain() {
            if (wip.getAndIncrement() != 0) {
                return;
            }
            int missed = 1;
            while (true) {
                long r = requested.get();
                long emitted = 0L;

                while (emitted != r) {
                    if (cancelled.get()) {
                        queue.clear();
                        return;
                    }
                    Throwable err = terminalError.get();
                    if (err != null) {
                        cancelled.set(true);
                        queue.clear();
                        subscriber.onError(err);
                        return;
                    }
                    ChatModelStreamingEvent event = queue.poll();
                    if (event == null) {
                        break;
                    }
                    subscriber.onNext(event);
                    emitted++;
                }

                if (emitted != 0L && r != Long.MAX_VALUE) {
                    requested.addAndGet(-emitted);
                }

                if (cancelled.get()) {
                    queue.clear();
                    return;
                }
                Throwable err = terminalError.get();
                if (err != null) {
                    cancelled.set(true);
                    queue.clear();
                    subscriber.onError(err);
                    return;
                }
                if (producerDone.get() && queue.isEmpty()) {
                    cancelled.set(true);
                    subscriber.onComplete();
                    return;
                }

                missed = wip.addAndGet(-missed);
                if (missed == 0) {
                    return;
                }
            }
        }

        private static void addDemandCapMax(AtomicLong current, long n) {
            while (true) {
                long r = current.get();
                if (r == Long.MAX_VALUE) {
                    return;
                }
                long u = r + n;
                if (u < 0L) {
                    u = Long.MAX_VALUE;
                }
                if (current.compareAndSet(r, u)) {
                    return;
                }
            }
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
            throw new IllegalStateException("LibArgusStreamingChatModel is already closed");
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
        private boolean returnThinking = false;
        private String modelName;
        private ChatRequestParameters defaultRequestParameters;
        private final List<ChatModelListener> listeners = new ArrayList<>();
        private final Set<Capability> supportedCapabilities = new LinkedHashSet<>();
        private Executor asyncExecutor;

        private Path mmprojPath;
        private boolean multimodalUseGpu = true;
        private Path allowedMediaRoot;
        private boolean allowLocalFileMedia = false;
        private long maxMediaBytes = LibArgusMediaSupport.DEFAULT_MAX_MEDIA_BYTES;
        private float videoFps = LibArgusMediaSupport.DEFAULT_VIDEO_FPS;
        private long videoTimestampIntervalMs = LibArgusMediaSupport.DEFAULT_VIDEO_TIMESTAMP_INTERVAL_MS;
        private int maxVideoFrames = LibArgusMediaSupport.DEFAULT_MAX_VIDEO_FRAMES;

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

        public Builder returnThinking(boolean returnThinking) {
            this.returnThinking = returnThinking;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder defaultRequestParameters(ChatRequestParameters defaultRequestParameters) {
            this.defaultRequestParameters = Objects.requireNonNull(defaultRequestParameters, "defaultRequestParameters");
            return this;
        }

        public Builder listeners(Collection<ChatModelListener> listeners) {
            Objects.requireNonNull(listeners, "listeners");
            this.listeners.clear();
            for (ChatModelListener listener : listeners) {
                if (listener != null) {
                    this.listeners.add(listener);
                }
            }
            return this;
        }

        public Builder addListener(ChatModelListener listener) {
            this.listeners.add(Objects.requireNonNull(listener, "listener"));
            return this;
        }

        public Builder supportedCapabilities(Collection<Capability> capabilities) {
            Objects.requireNonNull(capabilities, "capabilities");
            this.supportedCapabilities.clear();
            for (Capability cap : capabilities) {
                if (cap != null) {
                    this.supportedCapabilities.add(cap);
                }
            }
            return this;
        }

        public Builder supportedCapabilities(Capability... capabilities) {
            Objects.requireNonNull(capabilities, "capabilities");
            return supportedCapabilities(Arrays.asList(capabilities));
        }

        /**
         * Configures an explicit {@link Executor} for reactive {@link #chat(ChatRequest)}
         * ({@link Flow.Publisher}).
         *
         * <p>Never defaults to {@code ForkJoinPool.commonPool()} so GPU thread-local driver handles
         * (e.g. {@code cublasHandle_t}) remain pinned to caller-managed platform worker pools.
         */
        public Builder asyncExecutor(Executor asyncExecutor) {
            this.asyncExecutor = asyncExecutor;
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

        public Builder videoFps(float videoFps) {
            if (!Float.isFinite(videoFps) || videoFps <= 0.0f) {
                throw new IllegalArgumentException("videoFps must be finite and > 0 (got " + videoFps + ")");
            }
            this.videoFps = videoFps;
            return this;
        }

        public Builder videoTimestampIntervalMs(long videoTimestampIntervalMs) {
            if (videoTimestampIntervalMs < 0L) {
                throw new IllegalArgumentException("videoTimestampIntervalMs must be >= 0 (got " + videoTimestampIntervalMs + ")");
            }
            this.videoTimestampIntervalMs = videoTimestampIntervalMs;
            return this;
        }

        public Builder maxVideoFrames(int maxVideoFrames) {
            ArgusCommonValidation.checkPositive(maxVideoFrames, "maxVideoFrames");
            this.maxVideoFrames = maxVideoFrames;
            return this;
        }

        private ChatRequestParameters buildDefaultRequestParameters() {
            ChatRequestParameters base = defaultRequestParameters != null
                ? defaultRequestParameters
                : ChatRequestParameters.builder().build();
            if (modelName != null && base.modelName() == null) {
                return base.overrideWith(ChatRequestParameters.builder().modelName(modelName).build());
            }
            return base;
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

        public LibArgusStreamingChatModel build() {
            return new LibArgusStreamingChatModel(this);
        }
    }
}
