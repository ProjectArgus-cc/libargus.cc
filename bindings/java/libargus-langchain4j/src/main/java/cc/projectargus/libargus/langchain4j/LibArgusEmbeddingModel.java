package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingInput;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.embedding.response.EmbeddingResponseMetadata;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process, GPU-accelerated LangChain4j {@link EmbeddingModel} backed by {@code libargus}.
 *
 * <p>Defaults to bidirectional non-causal attention ({@code causalAttn = false}) with
 * {@code uBatch = contextLength} so full-window BERT, {@code EmbeddingGemma 2}, and
 * {@code jina-embeddings-v3} models evaluate in a single pass, supports {@link EmbeddingInputType}
 * asymmetric query/document prefixing and Matryoshka {@link EmbeddingRequest#dimensions()} slicing,
 * and applies L2 normalization by default.
 */
public final class LibArgusEmbeddingModel implements EmbeddingModel, AutoCloseable {

    /**
     * Metadata key on {@link TextSegment#metadata()} used to select {@code queryPrefix} ({@code "query"})
     * vs. {@code documentPrefix} ({@code "document"}) when both are configured.
     */
    public static final String METADATA_KEY_EMBEDDING_TYPE = "argus.embedding_type";

    private static final Set<EmbeddingParameter<?>> SUPPORTED_PARAMETERS = Set.of(
        EmbeddingRequestParameters.MODEL_NAME,
        EmbeddingRequestParameters.DIMENSIONS,
        EmbeddingRequestParameters.INPUT_TYPE
    );

    private final ArgusModel model;
    private final boolean ownsModel;
    private final ArgusContext context;
    private final int dimension;
    private final boolean normalize;
    private final String queryPrefix;
    private final String documentPrefix;
    private final String modelName;
    private final List<EmbeddingModelListener> listeners;
    private final Executor asyncExecutor;
    private final ThreadLocal<List<TextSegment>> activeSegmentsOverride = new ThreadLocal<>();
    private final ReentrantLock executionLock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusEmbeddingModel(Builder builder) {
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
        try {
            ArgusContextConfig ctxConfig = builder.buildContextConfig();
            createdContext = ArgusContext.init(resolvedModel, ctxConfig);
            this.dimension = createdContext.nEmbd();
        } catch (Throwable t) {
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
        this.normalize = builder.normalize;
        this.queryPrefix = builder.queryPrefix;
        this.documentPrefix = builder.documentPrefix;
        this.modelName = resolveInitialModelName(builder.modelName, resolvedModel);
        this.listeners = List.copyOf(builder.listeners);
        this.asyncExecutor = builder.asyncExecutor;
    }

    private static String resolveInitialModelName(String configuredName, ArgusModel model) {
        if (configuredName != null && !configuredName.isEmpty()) {
            return configuredName;
        }
        try {
            String ggufName = model.getMetadataValue("general.name");
            if (ggufName != null && !ggufName.isEmpty()) {
                return ggufName;
            }
        } catch (RuntimeException ignored) {
        }
        return "libargus";
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public String modelName() {
        return modelName;
    }

    @Override
    public List<EmbeddingModelListener> listeners() {
        return listeners;
    }

    @Override
    public Set<EmbeddingParameter<?>> supportedParameters() {
        return SUPPORTED_PARAMETERS;
    }

    @Override
    public Response<Embedding> embed(String text) {
        Objects.requireNonNull(text, "text");
        ensureOpen();
        EmbeddingResponse resp = embed(
            EmbeddingRequest.builder()
                .input(text)
                .inputType(EmbeddingInputType.QUERY)
                .build()
        );
        return Response.from(resp.embeddings().get(0), resp.tokenUsage());
    }

    @Override
    public Response<Embedding> embed(TextSegment textSegment) {
        Objects.requireNonNull(textSegment, "textSegment");
        ensureOpen();
        Response<List<Embedding>> batch = embedAll(List.of(textSegment));
        return Response.from(batch.content().get(0), batch.tokenUsage(), batch.finishReason());
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        Objects.requireNonNull(textSegments, "textSegments");
        ensureOpen();
        if (textSegments.isEmpty()) {
            return Response.from(Collections.emptyList(), new TokenUsage(0, 0));
        }
        for (int i = 0; i < textSegments.size(); i++) {
            Objects.requireNonNull(textSegments.get(i), "textSegments[" + i + "]");
        }

        activeSegmentsOverride.set(textSegments);
        try {
            EmbeddingResponse resp = embed(
                EmbeddingRequest.builder()
                    .textSegments(textSegments)
                    .build()
            );
            return Response.from(resp.embeddings(), resp.tokenUsage());
        } finally {
            activeSegmentsOverride.remove();
        }
    }

    @Override
    public EmbeddingResponse doEmbed(EmbeddingRequest embeddingRequest) {
        Objects.requireNonNull(embeddingRequest, "embeddingRequest");
        ensureOpen();

        List<EmbeddingInput> inputs = embeddingRequest.inputs();
        if (inputs == null || inputs.isEmpty()) {
            return EmbeddingResponse.builder()
                .embeddings(Collections.emptyList())
                .metadata(EmbeddingResponseMetadata.builder()
                    .modelName(resolveEffectiveModelName(embeddingRequest))
                    .tokenUsage(new TokenUsage(0, 0))
                    .build())
                .build();
        }

        int outDim = dimension;
        Integer requestedDim = embeddingRequest.dimensions();
        if (requestedDim != null) {
            if (requestedDim <= 0 || requestedDim > dimension) {
                throw new IllegalArgumentException(
                    "Requested embedding dimensions (" + requestedDim +
                    ") must be in [1, " + dimension + "]"
                );
            }
            outDim = requestedDim;
        }

        List<TextSegment> segmentOverride = activeSegmentsOverride.get();
        boolean hasSegmentOverride = segmentOverride != null && segmentOverride.size() == inputs.size();
        EmbeddingInputType requestInputType = embeddingRequest.inputType();

        List<Embedding> embeddings = new ArrayList<>(inputs.size());
        float[] reusableOut = new float[dimension];
        int totalInputTokens = 0;

        executionLock.lock();
        try {
            ensureOpen();
            for (int i = 0; i < inputs.size(); i++) {
                EmbeddingInput input = Objects.requireNonNull(inputs.get(i), "inputs[" + i + "]");
                TextSegment overrideSeg = hasSegmentOverride ? segmentOverride.get(i) : null;
                String rawText = overrideSeg != null && overrideSeg.text() != null
                    ? overrideSeg.text()
                    : input.text();
                String framed = applyPrefix(rawText, overrideSeg, requestInputType);
                if (framed.isEmpty()) {
                    throw new IllegalArgumentException("Cannot extract embeddings for empty text input at index " + i);
                }
                int[] tokens = context.tokenize(framed, true);
                if (tokens.length == 0) {
                    throw new IllegalStateException("Tokenization produced 0 tokens for input at index " + i);
                }
                totalInputTokens += tokens.length;

                // argus_eval_embeddings natively prunes slot 0 [0, -1) and resets sampler history under ctx->mtx
                context.evalEmbeddings(tokens, tokens.length, 0, reusableOut);

                float[] vec = (outDim == dimension)
                    ? reusableOut.clone()
                    : Arrays.copyOf(reusableOut, outDim);
                if (normalize) {
                    l2NormalizeInPlace(vec);
                }
                embeddings.add(Embedding.from(vec));
            }
        } finally {
            executionLock.unlock();
        }

        EmbeddingResponseMetadata meta = EmbeddingResponseMetadata.builder()
            .modelName(resolveEffectiveModelName(embeddingRequest))
            .tokenUsage(new TokenUsage(totalInputTokens, 0))
            .build();
        return EmbeddingResponse.builder()
            .embeddings(embeddings)
            .metadata(meta)
            .build();
    }

    @Override
    public CompletableFuture<EmbeddingResponse> doEmbedAsync(EmbeddingRequest embeddingRequest) {
        if (asyncExecutor == null) {
            return EmbeddingModel.super.doEmbedAsync(embeddingRequest);
        }
        Objects.requireNonNull(embeddingRequest, "embeddingRequest");
        ensureOpen();
        CompletableFuture<EmbeddingResponse> future = new CompletableFuture<>();
        try {
            asyncExecutor.execute(() -> {
                if (future.isCancelled()) {
                    return;
                }
                try {
                    future.complete(doEmbed(embeddingRequest));
                } catch (Throwable t) {
                    future.completeExceptionally(t);
                }
            });
        } catch (Throwable submitError) {
            future.completeExceptionally(submitError);
        }
        return future;
    }

    private String resolveEffectiveModelName(EmbeddingRequest request) {
        if (request != null && request.modelName() != null && !request.modelName().isEmpty()) {
            return request.modelName();
        }
        return modelName;
    }

    private String applyPrefix(String rawText, TextSegment overrideSegment, EmbeddingInputType requestInputType) {
        String raw = rawText != null ? rawText : "";
        if (queryPrefix == null && documentPrefix == null) {
            return raw;
        }
        String metaType = (overrideSegment != null && overrideSegment.metadata() != null)
            ? overrideSegment.metadata().getString(METADATA_KEY_EMBEDDING_TYPE)
            : null;
        if (metaType != null) {
            if ("query".equalsIgnoreCase(metaType) && queryPrefix != null) {
                return queryPrefix + raw;
            }
            if ("document".equalsIgnoreCase(metaType) && documentPrefix != null) {
                return documentPrefix + raw;
            }
        }
        if (requestInputType == EmbeddingInputType.QUERY && queryPrefix != null) {
            return queryPrefix + raw;
        }
        if (requestInputType == EmbeddingInputType.DOCUMENT && documentPrefix != null) {
            return documentPrefix + raw;
        }
        if (documentPrefix != null) {
            return documentPrefix + raw;
        }
        if (queryPrefix != null) {
            return queryPrefix + raw;
        }
        return raw;
    }

    private static void l2NormalizeInPlace(float[] vec) {
        double sumSq = 0.0;
        for (float v : vec) {
            if (Float.isFinite(v)) {
                sumSq += (double) v * (double) v;
            }
        }
        if (sumSq > 0.0) {
            float invNorm = (float) (1.0 / Math.sqrt(sumSq));
            for (int i = 0; i < vec.length; i++) {
                vec[i] = Float.isFinite(vec[i]) ? vec[i] * invNorm : 0.0f;
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
            throw new IllegalStateException("LibArgusEmbeddingModel is already closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        executionLock.lock();
        try {
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
        private int contextLength = 512;
        private int cpuThreads = ArgusCommonValidation.defaultCpuThreads();
        private int uBatch = 0;
        private Boolean causalAttn = Boolean.FALSE;

        private boolean normalize = true;
        private String queryPrefix;
        private String documentPrefix;
        private String modelName;
        private final List<EmbeddingModelListener> listeners = new ArrayList<>();
        private Executor asyncExecutor;

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

        public Builder uBatch(int uBatch) {
            ArgusCommonValidation.checkNonNegative(uBatch, "uBatch");
            this.uBatch = uBatch;
            return this;
        }

        public Builder causalAttn(Boolean causalAttn) {
            this.causalAttn = causalAttn;
            return this;
        }

        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder queryPrefix(String queryPrefix) {
            this.queryPrefix = queryPrefix;
            return this;
        }

        public Builder documentPrefix(String documentPrefix) {
            this.documentPrefix = documentPrefix;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder listeners(Collection<EmbeddingModelListener> listeners) {
            Objects.requireNonNull(listeners, "listeners");
            this.listeners.clear();
            for (EmbeddingModelListener listener : listeners) {
                if (listener != null) {
                    this.listeners.add(listener);
                }
            }
            return this;
        }

        public Builder addListener(EmbeddingModelListener listener) {
            this.listeners.add(Objects.requireNonNull(listener, "listener"));
            return this;
        }

        /**
         * Configures an explicit {@link Executor} for {@link #embedAsync(EmbeddingRequest)}.
         *
         * <p>Never defaults to {@code ForkJoinPool.commonPool()} so GPU thread-local driver handles
         * (e.g. {@code cublasHandle_t}) remain pinned to caller-managed platform worker pools.
         */
        public Builder asyncExecutor(Executor asyncExecutor) {
            this.asyncExecutor = asyncExecutor;
            return this;
        }

        private ArgusContextConfig buildContextConfig() {
            if (contextConfig != null) {
                return contextConfig;
            }
            int effectiveUBatch = uBatch;
            if (Boolean.FALSE.equals(causalAttn) && effectiveUBatch == 0) {
                effectiveUBatch = contextLength;
            }
            return new ArgusContextConfig.Builder(contextLength)
                .cpuThreads(cpuThreads)
                .uBatch(effectiveUBatch)
                .embeddings(true)
                .causalAttn(causalAttn)
                .build();
        }

        public LibArgusEmbeddingModel build() {
            return new LibArgusEmbeddingModel(this);
        }
    }
}
