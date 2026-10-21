# Java Developer Cookbook & Quickstart Guide

`libargus` provides an idiomatic, memory-safe, `AutoCloseable` Java API across both **JDK 22+ Project Panama FFM** (`:libargus-core`) and **JDK 17+ / Android JNI** (`:libargus-jni`).

Because both bindings expose the same package (`cc.projectargus.libargus.*`) and identical Arena-free high-level method signatures, **all standard Java examples below compile and run unmodified on both `libargus-core` and `libargus-jni`**.

---

## 1. Native Library Resolution Hierarchy

When `ArgusBackend` or any `libargus` resource class is initialized, [`ArgusNativeLoader`](../bindings/java/libargus-common/src/main/java/cc/projectargus/libargus/internal/ArgusNativeLoader.java) resolves the native `libargus` binary in the following order:

1. **Explicit Path Property:** `-Dcc.projectargus.libargus.path=/abs/path/to/libargus.so` (or directory containing it).
2. **Classpath SPI Auto-Extraction:** Automatically extracts the highest-priority matching [`NativeLibraryProvider`](../bindings/java/libargus-common/src/main/java/cc/projectargus/libargus/spi/NativeLibraryProvider.java) (`CUDA = 100`, `Metal = 100`, `ROCm = 90`, `Vulkan = 80`, `CPU = 10`) into `$TMPDIR/argus_native_cache_<user>` (overrideable via `-Dcc.projectargus.libargus.nativeDir=/custom/dir` or `LIBARGUS_NATIVE_DIR`).
3. **System Library Path:** `System.loadLibrary("argus")` (used automatically by Android APK `jniLibs` and system-wide installations).
4. **Local Build Directory Fallback:** Searches `./build/lib` and `./build/bin` up to 3 parent directories for local CMake development.

---

## 2. Streaming Text Generation (`generate` & `generateUntil`)

### 2.1 High-Level Streaming & Early-Stopping Predicates
`ArgusContext.generate(...)` and `ArgusContext.generateUntil(...)` handle prompt tokenization, prefill batching, sampler priming, and RFC 3629 stateful multi-byte UTF-8 decoding (preventing split CJK characters or 4-byte emojis from corrupting into `U+FFFD`):

```java
import cc.projectargus.libargus.*;
import java.nio.file.Path;

public class TextGenerationExample {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (ArgusModel model = ArgusModel.load(Path.of("models/llama-3-8b.gguf"), 99, true)) {
            ArgusContextConfig config = new ArgusContextConfig.Builder(4096)
                .cpuThreads(8)
                .typeK(ArgusContextConfig.KV_TYPE_Q4_0)
                .typeV(ArgusContextConfig.KV_TYPE_Q4_0)
                .build();

            try (ArgusContext context = ArgusContext.init(model, config)) {
                ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder()
                    .temperature(0.7f)
                    .topP(0.90f)
                    .minP(0.05f)
                    .repeatPenalty(1.1f)
                    .seed(42)
                    .build();

                // 1. Stream all tokens up to maxTokens (or EOG) via Consumer<String>:
                String fullText = context.generate(
                    "Explain quantum entanglement in three sentences:",
                    256,
                    sampler,
                    System.out::print
                );

                // 2. Or halt early from inside the callback via generateUntil (Predicate<String>):
                String firstParagraph = context.generateUntil(
                    "Write a short essay on operating systems:",
                    512,
                    sampler,
                    piece -> {
                        System.out.print(piece);
                        return !piece.contains("\n\n"); // Return false to stop generation immediately
                    }
                );
            }
        } finally {
            ArgusBackend.free();
        }
    }
}
```

### 2.2 Cross-Thread Cooperative Cancellation (`ArgusAbortFlag`)
Pass an [`ArgusAbortFlag`](../bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/ArgusAbortFlag.java) to cancel long prompt prefills or ongoing generation from another thread (such as a UI cancel button, HTTP timeout handler, or virtual thread):

```java
try (ArgusContext context = ArgusContext.init(model, config);
     ArgusAbortFlag abortFlag = new ArgusAbortFlag()) {

    // From a timeout scheduler or cancellation thread:
    // abortFlag.abort();

    String partial = context.generate(
        largePrompt,
        1024,
        sampler,
        System.out::print,
        abortFlag
    );
}
```

---

## 3. Low-Level Autoregressive Loop, Persistent Samplers & Logit Bias

For custom token-by-token orchestration, `ArgusContext` exposes direct batch decoding, persistent per-slot sampler chains, decoupled penalty priming, and logit steering:

```java
ArgusSamplerConfig samplerConfig = new ArgusSamplerConfig.Builder()
    .temperature(0.7f)
    .seed(42)
    .topP(0.90f)
    .minP(0.05f)
    .topK(40)
    .repeatPenalty(1.1f)
    .repeatLastN(64)
    .dry(0.8f, 1.75f, 2, -1) // DRY (Don't Repeat Yourself) n-gram suppression
    .build();

int[] promptTokens = context.tokenize("Translate to French: Hello world", true);
context.decodeBatch(promptTokens, promptTokens.length, 0, 0, true);
context.primeSampler(0, promptTokens, promptTokens.length);

// Optional logit bias arrays (e.g., banning specific token IDs with -Float.MAX_VALUE)
int[] bannedTokens = new int[] { 151644, 151645 };
float[] bannedWeights = new float[] { -Float.MAX_VALUE, -Float.MAX_VALUE };

int[] singleToken = new int[1];
int currentPos = promptTokens.length;

for (int step = 0; step < 128; step++) {
    int token = context.sampleTokenWithBias(0, samplerConfig, bannedTokens, bannedWeights, bannedTokens.length);
    if (context.vocabIsEog(token)) {
        break;
    }
    System.out.print(context.tokenToPiece(token));

    singleToken[0] = token;
    context.decodeBatch(singleToken, 1, currentPos++, 0, true);
}
```

> [!TIP]
> **Zero-Allocation Panama FFM Hot Path:** When running on `:libargus-core` (JDK 22+), you can pre-allocate `MemorySegment` buffers (`ArgusLayouts.TOKEN_BATCH` and `ArgusLayouts.LOGIT_BIAS`) once outside the loop and call `context.decodeBatch(batchSeg)` and `context.sampleTokenWithBias(0, samplerConfig, biasSeg, count)` for **zero per-token Java heap allocations**.

---

## 4. Semantic Text Embeddings

Extract dense float embedding vectors from dedicated embedding models (e.g., `jina-embeddings-v3`, `EmbeddingGemma2`, `nomic-embed-text`):

```java
import cc.projectargus.libargus.*;
import java.nio.file.Path;

public class EmbeddingsExample {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (ArgusModel model = ArgusModel.load(Path.of("models/jina-embeddings-v3-Q4_K_M.gguf"), 99, true)) {
            ArgusContextConfig config = new ArgusContextConfig.Builder(512)
                .cpuThreads(4)
                .embeddings(true)
                .causalAttn(false) // Bidirectional non-causal attention for embedding architectures
                .build();

            try (ArgusContext context = ArgusContext.init(model, config)) {
                // 1. One-line extraction into a new float[]:
                float[] vector = context.extractEmbeddings("Retrieve semantic vector for this sentence.");
                System.out.println("Embedding dimension: " + vector.length);

                // 2. Or reuse a caller-allocated float[] across pre-tokenized batches:
                int[] tokens = context.tokenize("Reusable buffer embedding query", true);
                float[] reusableOut = new float[context.nEmbd()];
                int written = context.evalEmbeddings(tokens, tokens.length, 0, reusableOut);
            }
        } finally {
            ArgusBackend.free();
        }
    }
}
```

---

## 5. Multimodal Vision, Audio & Frame-by-Frame Video

### 5.1 Vision & Audio Prompting (`ArgusMultimodalContext` & `ArgusBitmap`)
Use a multimodal GGUF model and its companion projector (`mmproj`) to evaluate images or audio clips inline with text prompts using the `<__media__>` marker:

```java
import cc.projectargus.libargus.*;
import java.nio.file.Path;
import java.util.List;

public class VisionPromptExample {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (ArgusModel model = ArgusModel.load(Path.of("models/qwen2-vl-7b-it.gguf"), 99, true);
             ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(8192).build());
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(
                 model, Path.of("models/qwen2-vl-7b-it.mmproj"), 4, true)) {

            try (ArgusBitmap image = ArgusBitmap.loadFile(mctx, Path.of("media/diagram.png"), false);
                 ArgusInputChunks chunks = mctx.tokenize(
                     "<__media__>\nSummarize the architecture in this diagram.", true, List.of(image))) {

                // 1. Project image patches & evaluate multimodal prompt chunks on sequence slot 0
                int currentPos = context.evalMultimodalChunks(mctx, chunks, 0, 0, 1024, true);

                // 2. Sample and stream the multimodal response starting from currentPos
                ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder().temperature(0.2f).build();
                int[] nextTokenBuf = new int[1];

                for (int step = 0; step < 256; step++) {
                    int token = context.sampleToken(0, sampler);
                    if (context.vocabIsEog(token)) {
                        break;
                    }
                    System.out.print(context.tokenToPiece(token));
                    nextTokenBuf[0] = token;
                    context.decodeBatch(nextTokenBuf, 1, currentPos++, 0, true);
                }
            }
        } finally {
            ArgusBackend.free();
        }
    }
}
```

### 5.2 Frame-by-Frame Video Stream Processing (`ArgusVideo` & `ArgusVideoItem`)
Stream video files or in-memory buffers through FFmpeg into recycled [`ArgusVideoItem`](../bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/ArgusVideoItem.java) carriers (requires `ffmpeg` and `ffprobe` on `PATH`):

```java
try (ArgusVideo video = ArgusVideo.loadFile(mctx, Path.of("media/clip.mp4"), 2.0f, 5000);
     ArgusVideoItem item = new ArgusVideoItem()) {
    while (video.readNext(item)) {
        if (item.bitmap() != null) {
            ArgusBitmap frame = item.bitmap(); // Owned by ArgusVideoItem until next readNext() or close()
            // Tokenize or batch frame...
        } else if (item.text() != null) {
            System.out.println("Timestamp marker: " + item.text()); // e.g. "[00m05s]"
        }
    }
}
```

---

## 6. Whisper Speech-to-Text (ASR) & Speech Synthesis (TTS)

### 6.1 Whisper Audio Transcription (`ArgusAudioContext`)
Transcribe 16 kHz mono `float[]` PCM audio normalized to `[-1.0f, 1.0f]`:

```java
import cc.projectargus.libargus.*;
import java.nio.file.Path;

try (ArgusAudioContext whisper = ArgusAudioContext.init(Path.of("models/whisper-base.en.gguf"), 4, 99)) {
    float[] pcm16kHz = loadNormalized16kHzAudio();
    String transcript = whisper.transcribe(pcm16kHz, pcm16kHz.length, 4096);
    System.out.println("Transcript: " + transcript);
}
```

### 6.2 Speech-LLM Text-to-Speech (`synthesizeSpeech`)
Synthesize 24 kHz mono PCM audio using a Speech-LLM (such as OuteTTS) and a WavTokenizer vocoder GGUF model:

```java
try (ArgusModel ttsModel = ArgusModel.load(Path.of("models/outetts-0.2-500m.gguf"), 99, true);
     ArgusModel vocoder = ArgusModel.load(Path.of("models/wavtokenizer-large-75token.gguf"), 99, true);
     ArgusContext ttsContext = ArgusContext.init(ttsModel, new ArgusContextConfig.Builder(4096).build())) {

    int maxSamples = 24000 * 15; // Up to 15 seconds of 24kHz audio
    float[] pcmSamples = ttsContext.synthesizeSpeech(vocoder, "Welcome to Project Argus.", 42, maxSamples);
    System.out.println("Synthesized " + pcmSamples.length + " audio samples at 24kHz.");
}
```

---

## 7. Sequence Slot Forking, Prefix Rollback & Sliding-Window Eviction

### 7.1 Zero-Rollback Sequence Slot Forking (`forkSlot` / `freeSlot`)
Explore parallel reasoning branches (Tree-of-Thoughts, MCTS, or concurrent agent workers sharing a system prompt) or branch safely on hybrid recurrent architectures (`model.isHybrid() == true`) where partial KV rollback is unsupported:

```java
ArgusContextConfig config = new ArgusContextConfig.Builder(4096)
    .seqMax(1)
    .cloneSlots(4) // Reserves 4 dedicated worker slots for forkSlot()
    .kvUnified(true)
    .build();

try (ArgusContext ctx = ArgusContext.init(model, config)) {
    // 1. Prefill shared system prompt once on root slot 0
    int[] systemTokens = ctx.tokenize("System: You are a helpful assistant.\n", true);
    ctx.decodeBatch(systemTokens, systemTokens.length, 0, 0, false);

    // 2. Fork root slot 0 into an isolated worker slot (clones KV cache + sampler RNG state)
    int workerSlot = ctx.forkSlot(0);
    try {
        int[] userTokens = ctx.tokenize("User: Solve 2+2.", false);
        ctx.decodeBatch(userTokens, userTokens.length, ctx.getSeqPosMax(0) + 1, workerSlot, true);
        int token = ctx.sampleToken(workerSlot, sampler);
    } finally {
        // 3. Release worker slot back to the pool
        ctx.freeSlot(workerSlot);
    }
}
```

### 7.2 Sliding-Window Ring-Buffer Eviction (`clearCacheSlot` & `shiftCacheSlot`)
For infinite-horizon streaming on architectures that support KV position shifting (`context.canShift() == true`):

```java
if (context.canShift()) {
    int nKeep = 64;     // Keep pinned system prompt prefix [0, 64)
    int nDiscard = 256; // Evict oldest conversation window [64, 320)
    context.clearCacheSlot(0, nKeep, nKeep + nDiscard);
    context.shiftCacheSlot(0, nKeep + nDiscard, -1, -nDiscard);
    // Continue decoding at startPos = context.getSeqPosMax(0) + 1
}
```

---

## 8. Model Introspection, VRAM Budgeting, Logging & Telemetry

### 8.1 Pre-Allocation VRAM Budgeting & GGUF Metadata
```java
long weightBytes = model.modelSize();
long kvPerToken = model.kvBytesPerToken(ArgusContextConfig.KV_TYPE_Q4_0, ArgusContextConfig.KV_TYPE_Q4_0);
long totalVramFor32k = model.estimateVramBytes(32768, ArgusContextConfig.KV_TYPE_Q4_0, ArgusContextConfig.KV_TYPE_Q4_0);

String arch = model.getMetadataValue("general.architecture");
Map<String, String> allMeta = model.getMetadataMap();
```

### 8.2 Native Logging Control & Custom Log Routing
```java
// Adjust global threshold (defaults to WARN; also configurable via LIBARGUS_LOG_LEVEL=INFO)
ArgusBackend.setLogLevel(ArgusLogLevel.INFO);

// Route native logs into SLF4J / Logback / Android Log:
ArgusBackend.setLogCallback((level, message) -> {
    if (level == ArgusLogLevel.ERROR) {
        System.err.print("[libargus] " + message);
    }
});
```

### 8.3 Hardware-Synchronized Performance Telemetry (`ArgusPerfTimings`)
```java
ArgusPerfTimings perf = context.getPerfTimings();
System.out.printf("Prefill: %d tokens (%.2f tok/s) | Decode: %d tokens (%.2f tok/s)%n",
    perf.nPromptEval(), perf.promptTokensPerSecond(),
    perf.nEval(), perf.evalTokensPerSecond());
context.resetPerfTimings();
```
