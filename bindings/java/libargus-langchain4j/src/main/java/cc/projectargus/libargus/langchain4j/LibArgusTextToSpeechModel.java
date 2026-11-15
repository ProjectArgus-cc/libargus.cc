package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusContext;
import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.model.audio.TextToSpeechModel;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import dev.langchain4j.model.audio.TextToSpeechResponse;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process, GPU-accelerated LangChain4j {@link TextToSpeechModel} backed by {@code libargus}'s
 * Speech-LLM + WavTokenizer vocoder DSP pipeline ({@link ArgusContext#synthesizeSpeech}).
 *
 * <p>Emits canonical 16-bit signed little-endian RIFF/WAVE ({@code audio/wav}) payloads inside
 * {@link Audio#binaryData()}, ready for playback or direct round-tripping into
 * {@link LibArgusAudioTranscriptionModel}.
 */
public final class LibArgusTextToSpeechModel implements TextToSpeechModel, AutoCloseable {

    private static final int DEFAULT_VOICE_SEED = 42;
    private static final int DEFAULT_MAX_SAMPLES = LibArgusAudioCodec.DEFAULT_TTS_SAMPLE_RATE_HZ * 30; // 30s @ 24kHz

    private final ArgusModel model;
    private final boolean ownsModel;
    private final ArgusModel vocoderModel;
    private final boolean ownsVocoderModel;
    private final ArgusContext context;
    private final int defaultVoiceSeed;
    private final Map<String, Integer> namedVoiceSeeds;
    private final int maxSamples;
    private final int sampleRate;
    private final ReentrantLock executionLock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusTextToSpeechModel(Builder builder) {
        ArgusBackend.init();

        ArgusModel loadedModel = null;
        boolean modelOwned = false;
        ArgusModel loadedVocoder = null;
        boolean vocoderOwned = false;
        ArgusContext createdContext = null;

        try {
            if (builder.model != null) {
                loadedModel = builder.model;
                modelOwned = false;
            } else if (builder.modelPath != null) {
                loadedModel = ArgusModel.load(builder.modelPath, builder.gpuLayers, builder.useMlock);
                modelOwned = true;
            } else if (builder.modelFd != null) {
                loadedModel = ArgusModel.loadFd(builder.modelFd, builder.gpuLayers, builder.useMlock);
                modelOwned = true;
            } else {
                throw new IllegalArgumentException(
                    "Builder requires Speech-LLM base model via model(ArgusModel), modelPath(Path), or modelFd(int)"
                );
            }

            if (builder.vocoderModel != null) {
                loadedVocoder = builder.vocoderModel;
                vocoderOwned = false;
            } else if (builder.vocoderModelPath != null) {
                loadedVocoder = ArgusModel.load(builder.vocoderModelPath, builder.vocoderGpuLayers, builder.useMlock);
                vocoderOwned = true;
            } else if (builder.vocoderModelFd != null) {
                loadedVocoder = ArgusModel.loadFd(builder.vocoderModelFd, builder.vocoderGpuLayers, builder.useMlock);
                vocoderOwned = true;
            } else {
                throw new IllegalArgumentException(
                    "Builder requires WavTokenizer vocoder model via vocoderModel(ArgusModel), vocoderModelPath(Path), or vocoderModelFd(int)"
                );
            }

            createdContext = ArgusContext.init(loadedModel, builder.buildContextConfig());
        } catch (Throwable t) {
            if (createdContext != null) {
                try { createdContext.close(); } catch (Throwable ignored) {}
            }
            if (vocoderOwned && loadedVocoder != null) {
                try { loadedVocoder.close(); } catch (Throwable ignored) {}
            }
            if (modelOwned && loadedModel != null) {
                try { loadedModel.close(); } catch (Throwable ignored) {}
            }
            throw t;
        }

        this.model = loadedModel;
        this.ownsModel = modelOwned;
        this.vocoderModel = loadedVocoder;
        this.ownsVocoderModel = vocoderOwned;
        this.context = createdContext;
        this.defaultVoiceSeed = builder.defaultVoiceSeed;
        this.namedVoiceSeeds = Collections.unmodifiableMap(new LinkedHashMap<>(builder.namedVoiceSeeds));
        this.maxSamples = builder.maxSamples;
        this.sampleRate = builder.sampleRate;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public TextToSpeechResponse synthesize(TextToSpeechRequest request) {
        Objects.requireNonNull(request, "request");
        String text = request.text();
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("TextToSpeechRequest.text() cannot be null or empty");
        }
        int seed = resolveVoiceSeed(request.voice());
        float[] pcm = synthesizePcm(text, seed);
        byte[] wavBytes = LibArgusAudioCodec.encodeMonoPcm16Wav(pcm, sampleRate);
        Audio audio = Audio.builder()
            .binaryData(wavBytes)
            .mimeType("audio/wav")
            .build();
        return TextToSpeechResponse.from(audio);
    }

    /**
     * Direct zero-container convenience entrypoint returning raw normalized {@code [-1.0f, 1.0f]}
     * mono PCM samples synthesized by the Speech-LLM + WavTokenizer vocoder.
     */
    public float[] synthesizePcm(String text, int voiceSeed) {
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) {
            throw new IllegalArgumentException("text cannot be empty");
        }
        ensureOpen();

        executionLock.lock();
        try {
            ensureOpen();
            return context.synthesizeSpeech(vocoderModel, text, voiceSeed, maxSamples);
        } finally {
            executionLock.unlock();
        }
    }

    int resolveVoiceSeed(String voice) {
        if (voice == null) {
            return defaultVoiceSeed;
        }
        String trimmed = voice.trim();
        if (trimmed.isEmpty()) {
            return defaultVoiceSeed;
        }
        Integer mapped = namedVoiceSeeds.get(trimmed.toLowerCase(Locale.ROOT));
        if (mapped != null) {
            return mapped;
        }
        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException ignored) {
            return trimmed.hashCode() & 0x7FFFFFFF;
        }
    }

    public int sampleRate() {
        return sampleRate;
    }

    public ArgusModel getModel() {
        return model;
    }

    public ArgusModel getVocoderModel() {
        return vocoderModel;
    }

    public ArgusContext getContext() {
        return context;
    }

    public boolean isClosed() {
        return closed.get() || context.isClosed();
    }

    private void ensureOpen() {
        if (closed.get() || context.isClosed()) {
            throw new IllegalStateException("LibArgusTextToSpeechModel is already closed");
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
                try {
                    if (ownsVocoderModel) {
                        vocoderModel.close();
                    }
                } finally {
                    if (ownsModel) {
                        model.close();
                    }
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

        private ArgusModel vocoderModel;
        private Path vocoderModelPath;
        private Integer vocoderModelFd;
        private int vocoderGpuLayers = 99;

        private ArgusContextConfig contextConfig;
        private int contextLength = 2048;
        private int cpuThreads = ArgusCommonValidation.defaultCpuThreads();
        private int typeK = ArgusContextConfig.KV_TYPE_F16;
        private int typeV = ArgusContextConfig.KV_TYPE_F16;

        private int defaultVoiceSeed = DEFAULT_VOICE_SEED;
        private final Map<String, Integer> namedVoiceSeeds = new LinkedHashMap<>();
        private int maxSamples = DEFAULT_MAX_SAMPLES;
        private int sampleRate = LibArgusAudioCodec.DEFAULT_TTS_SAMPLE_RATE_HZ;

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
            this.vocoderGpuLayers = gpuLayers;
            return this;
        }

        public Builder useMlock(boolean useMlock) {
            this.useMlock = useMlock;
            return this;
        }

        public Builder vocoderModel(ArgusModel vocoderModel) {
            this.vocoderModel = Objects.requireNonNull(vocoderModel, "vocoderModel");
            this.vocoderModelPath = null;
            this.vocoderModelFd = null;
            return this;
        }

        public Builder vocoderModelPath(Path vocoderModelPath) {
            this.vocoderModelPath = Objects.requireNonNull(vocoderModelPath, "vocoderModelPath");
            this.vocoderModel = null;
            this.vocoderModelFd = null;
            return this;
        }

        public Builder vocoderModelFd(int fd) {
            if (fd < 0) {
                throw new IllegalArgumentException("Invalid file descriptor: " + fd);
            }
            this.vocoderModelFd = fd;
            this.vocoderModel = null;
            this.vocoderModelPath = null;
            return this;
        }

        public Builder vocoderGpuLayers(int vocoderGpuLayers) {
            this.vocoderGpuLayers = vocoderGpuLayers;
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

        public Builder defaultVoiceSeed(int defaultVoiceSeed) {
            this.defaultVoiceSeed = defaultVoiceSeed;
            return this;
        }

        public Builder voice(String voiceName, int voiceSeed) {
            Objects.requireNonNull(voiceName, "voiceName");
            String normalized = voiceName.trim().toLowerCase(Locale.ROOT);
            if (normalized.isEmpty()) {
                throw new IllegalArgumentException("voiceName cannot be blank");
            }
            this.namedVoiceSeeds.put(normalized, voiceSeed);
            return this;
        }

        public Builder maxSamples(int maxSamples) {
            ArgusCommonValidation.checkPositive(maxSamples, "maxSamples");
            this.maxSamples = maxSamples;
            return this;
        }

        public Builder sampleRate(int sampleRate) {
            ArgusCommonValidation.checkPositive(sampleRate, "sampleRate");
            this.sampleRate = sampleRate;
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

        public LibArgusTextToSpeechModel build() {
            return new LibArgusTextToSpeechModel(this);
        }
    }
}
