package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process, GPU-accelerated LangChain4j {@link EmbeddingModel} backed by {@code libargus}.
 *
 * <p>Defaults to bidirectional non-causal attention ({@code causalAttn = false}) with
 * {@code uBatch = contextLength} so full-window BERT, {@code EmbeddingGemma 2}, and
 * {@code jina-embeddings-v3} models evaluate in a single pass, and applies L2 normalization
 * by default.
 */
public final class LibArgusEmbeddingModel implements EmbeddingModel, AutoCloseable {

    /**
     * Metadata key on {@link TextSegment#metadata()} used to select {@code queryPrefix} ({@code "query"})
     * vs. {@code documentPrefix} ({@code "document"}) when both are configured.
     */
    public static final String METADATA_KEY_EMBEDDING_TYPE = "argus.embedding_type";

    private final ArgusModel model;
    private final boolean ownsModel;
    private final ArgusContext context;
    private final int dimension;
    private final boolean normalize;
    private final String queryPrefix;
    private final String documentPrefix;
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
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public Response<Embedding> embed(String text) {
        Objects.requireNonNull(text, "text");
        ensureOpen();
        Metadata meta = queryPrefix != null ? Metadata.from(METADATA_KEY_EMBEDDING_TYPE, "query") : new Metadata();
        Response<List<Embedding>> batch = embedAll(List.of(TextSegment.from(text, meta)));
        return Response.from(batch.content().get(0), batch.tokenUsage(), batch.finishReason());
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        Objects.requireNonNull(textSegments, "textSegments");
        ensureOpen();
        if (textSegments.isEmpty()) {
            return Response.from(Collections.emptyList(), new TokenUsage(0, 0));
        }

        List<Embedding> embeddings = new ArrayList<>(textSegments.size());
        float[] reusableOut = new float[dimension];
        int totalInputTokens = 0;

        executionLock.lock();
        try {
            ensureOpen();
            for (int i = 0; i < textSegments.size(); i++) {
                TextSegment segment = Objects.requireNonNull(textSegments.get(i), "textSegments[" + i + "]");
                String framed = applyPrefix(segment);
                if (framed.isEmpty()) {
                    throw new IllegalArgumentException("Cannot extract embeddings for empty text segment at index " + i);
                }
                int[] tokens = context.tokenize(framed, true);
                if (tokens.length == 0) {
                    throw new IllegalStateException("Tokenization produced 0 tokens for segment at index " + i);
                }
                totalInputTokens += tokens.length;

                // argus_eval_embeddings natively prunes slot 0 [0, -1) and resets sampler history under ctx->mtx
                context.evalEmbeddings(tokens, tokens.length, 0, reusableOut);

                float[] vec = reusableOut.clone();
                if (normalize) {
                    l2NormalizeInPlace(vec);
                }
                embeddings.add(Embedding.from(vec));
            }
        } finally {
            executionLock.unlock();
        }

        return Response.from(embeddings, new TokenUsage(totalInputTokens, 0));
    }

    private String applyPrefix(TextSegment segment) {
        String raw = segment.text() != null ? segment.text() : "";
        if (queryPrefix == null && documentPrefix == null) {
            return raw;
        }
        String type = segment.metadata() != null ? segment.metadata().getString(METADATA_KEY_EMBEDDING_TYPE) : null;
        if ("query".equalsIgnoreCase(type) && queryPrefix != null) {
            return queryPrefix + raw;
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
