package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusAudioContext;
import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.internal.ArgusCommonValidation;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.model.audio.AudioTranscriptionModel;
import dev.langchain4j.model.audio.AudioTranscriptionRequest;
import dev.langchain4j.model.audio.AudioTranscriptionResponse;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process, GPU-accelerated LangChain4j {@link AudioTranscriptionModel} backed by
 * {@code libargus}'s Whisper ASR engine ({@link ArgusAudioContext}).
 *
 * <p>Decodes RIFF/WAVE containers (8/16/24/32-bit PCM and 32-bit IEEE float, mono or multi-channel,
 * arbitrary sample rate resampled to 16 kHz) and raw PCM payloads ({@code audio/L16},
 * {@code audio/pcm}, {@code audio/f32le}) using a zero-dependency, Android API 26+-compatible
 * codec while enforcing {@code libargus}'s Zero-Trust media sandbox policy.
 */
public final class LibArgusAudioTranscriptionModel implements AudioTranscriptionModel, AutoCloseable {

    private static final int DEFAULT_MAX_OUTPUT_CHARS = 32_768;

    private final ArgusAudioContext audioContext;
    private final boolean ownsAudioContext;
    private final int maxOutputChars;
    private final boolean trimOutput;
    private final boolean padShortAudioToOneSecond;
    private final Path allowedMediaRoot;
    private final boolean allowLocalFileMedia;
    private final long maxMediaBytes;
    private final ReentrantLock executionLock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private LibArgusAudioTranscriptionModel(Builder builder) {
        ArgusBackend.init();

        if (builder.audioContext != null) {
            this.audioContext = builder.audioContext;
            this.ownsAudioContext = false;
        } else if (builder.modelPath != null) {
            this.audioContext = ArgusAudioContext.init(builder.modelPath, builder.cpuThreads, builder.gpuLayers);
            this.ownsAudioContext = true;
        } else {
            throw new IllegalArgumentException(
                "Builder requires either audioContext(ArgusAudioContext) or modelPath(Path)"
            );
        }

        this.maxOutputChars = builder.maxOutputChars;
        this.trimOutput = builder.trimOutput;
        this.padShortAudioToOneSecond = builder.padShortAudioToOneSecond;
        this.allowedMediaRoot = builder.allowedMediaRoot;
        this.allowLocalFileMedia = builder.allowLocalFileMedia;
        this.maxMediaBytes = builder.maxMediaBytes;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public AudioTranscriptionResponse transcribe(AudioTranscriptionRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();

        Audio audio = request.audio();
        byte[] rawAudioBytes = LibArgusMediaSupport.loadAudioBytes(
            audio,
            allowedMediaRoot,
            allowLocalFileMedia,
            maxMediaBytes,
            "AudioTranscriptionRequest"
        );
        float[] pcm16kHz = LibArgusAudioCodec.decodeToMono16kHzPcm(
            rawAudioBytes,
            audio.mimeType(),
            maxMediaBytes
        );
        String transcript = transcribe(pcm16kHz, pcm16kHz.length);
        return AudioTranscriptionResponse.from(transcript);
    }

    /**
     * Direct zero-container convenience entrypoint transcribing normalized 16 kHz mono PCM samples.
     */
    public String transcribe(float[] pcm16kHz) {
        Objects.requireNonNull(pcm16kHz, "pcm16kHz");
        return transcribe(pcm16kHz, pcm16kHz.length);
    }

    /**
     * Direct zero-container convenience entrypoint transcribing {@code numSamples} of 16 kHz mono PCM samples.
     */
    public String transcribe(float[] pcm16kHz, int numSamples) {
        Objects.requireNonNull(pcm16kHz, "pcm16kHz");
        if (numSamples <= 0 || numSamples > pcm16kHz.length) {
            throw new IllegalArgumentException(
                "numSamples (" + numSamples + ") must be in [1, " + pcm16kHz.length + "]"
            );
        }
        ensureOpen();

        final float[] effectivePcm;
        final int effectiveSamples;
        if (padShortAudioToOneSecond && numSamples < LibArgusAudioCodec.WHISPER_SAMPLE_RATE_HZ) {
            effectivePcm = Arrays.copyOf(pcm16kHz, LibArgusAudioCodec.WHISPER_SAMPLE_RATE_HZ);
            if (numSamples < pcm16kHz.length) {
                Arrays.fill(
                    effectivePcm,
                    numSamples,
                    Math.min(pcm16kHz.length, LibArgusAudioCodec.WHISPER_SAMPLE_RATE_HZ),
                    0.0f
                );
            }
            effectiveSamples = LibArgusAudioCodec.WHISPER_SAMPLE_RATE_HZ;
        } else {
            effectivePcm = pcm16kHz;
            effectiveSamples = numSamples;
        }

        executionLock.lock();
        try {
            ensureOpen();
            String raw = audioContext.transcribe(effectivePcm, effectiveSamples, maxOutputChars);
            if (raw == null) {
                return "";
            }
            return trimOutput ? raw.trim() : raw;
        } finally {
            executionLock.unlock();
        }
    }

    public ArgusAudioContext getAudioContext() {
        return audioContext;
    }

    public boolean isClosed() {
        return closed.get() || audioContext.isClosed();
    }

    private void ensureOpen() {
        if (closed.get() || audioContext.isClosed()) {
            throw new IllegalStateException("LibArgusAudioTranscriptionModel is already closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        executionLock.lock();
        try {
            if (ownsAudioContext) {
                audioContext.close();
            }
        } finally {
            executionLock.unlock();
        }
    }

    public static final class Builder {
        private ArgusAudioContext audioContext;
        private Path modelPath;
        private int cpuThreads = ArgusCommonValidation.defaultCpuThreads();
        private int gpuLayers = 99;
        private int maxOutputChars = DEFAULT_MAX_OUTPUT_CHARS;
        private boolean trimOutput = false;
        private boolean padShortAudioToOneSecond = true;
        private Path allowedMediaRoot;
        private boolean allowLocalFileMedia = false;
        private long maxMediaBytes = LibArgusMediaSupport.DEFAULT_MAX_MEDIA_BYTES;

        public Builder audioContext(ArgusAudioContext audioContext) {
            this.audioContext = Objects.requireNonNull(audioContext, "audioContext");
            this.modelPath = null;
            return this;
        }

        public Builder modelPath(Path modelPath) {
            this.modelPath = Objects.requireNonNull(modelPath, "modelPath");
            this.audioContext = null;
            return this;
        }

        public Builder cpuThreads(int cpuThreads) {
            ArgusCommonValidation.checkPositive(cpuThreads, "cpuThreads");
            this.cpuThreads = cpuThreads;
            return this;
        }

        public Builder gpuLayers(int gpuLayers) {
            this.gpuLayers = gpuLayers;
            return this;
        }

        public Builder maxOutputChars(int maxOutputChars) {
            ArgusCommonValidation.checkPositive(maxOutputChars, "maxOutputChars");
            this.maxOutputChars = maxOutputChars;
            return this;
        }

        public Builder trimOutput(boolean trimOutput) {
            this.trimOutput = trimOutput;
            return this;
        }

        /**
         * Zero-pads sub-1.0s clips with trailing silence up to 16,000 samples (1.0s at 16 kHz)
         * so {@code whisper.cpp} evaluates short spoken commands (e.g. {@code "Yes"}, {@code "Stop"})
         * instead of skipping sub-second windows. Enabled ({@code true}) by default.
         */
        public Builder padShortAudioToOneSecond(boolean padShortAudioToOneSecond) {
            this.padShortAudioToOneSecond = padShortAudioToOneSecond;
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

        public LibArgusAudioTranscriptionModel build() {
            return new LibArgusAudioTranscriptionModel(this);
        }
    }
}
