# libargus
## An unmanaged, zero-allocation native AI execution runtime consolidating Vision, Speech, and LLM compute pipelines behind a single Project Panama FFM boundary.

[![Release Pipeline](https://github.com/ProjectArgus-cc/libargus.cc/actions/workflows/release.yml/badge.svg)](https://github.com/ProjectArgus-cc/libargus.cc/actions/workflows/release.yml)
[![Maven Central](https://img.shields.io/maven-central/v/cc.projectargus/libargus-core.svg?label=Maven%20Central&color=blue)](https://central.sonatype.com/artifact/cc.projectargus/libargus-core)
[![JDK Target](https://img.shields.io/badge/JDK-22%2B%20Panama%20FFM-orange.svg)](https://openjdk.org/jeps/454)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **v1.6.4 Release — Sampler Continuation State Machine, Multimodal Mutex Serialization & Monotonic RNG Continuity**
> 
> * **Autoregressive Continuation State Machine:** Introduces `struct argus_pending_sample` decoupling uncommitted samples from committed KV history. Normal continuation commits tokens directly to history with zero chain resets, zero history replays, and zero RNG stream interruptions.
> * **Multimodal Context Synchronization:** Restores per-context mutex serialization (`std::lock_guard<std::mutex> lock(ctx->mtx)`) across `argus_eval_multimodal_chunks()`, eliminating data races against concurrent text decoding, sampling, and KV cache mutations.
> * **Monotonic RNG Stream Continuity:** Preserves and transfers active distribution sampler RNG states during partial KV rollbacks, mismatched token reconciliations, and truncations, preventing pseudorandom sequences from restarting to draw #0.
> * **Mismatched Sample Eviction:** Safely purges un-decoded speculative samples and rebuilds filter chains when decoded batch tokens deviate from sampled predictions.
> * **History & Pending Introspection:** Exposes zero-allocation C and Java Panama FFM APIs (`argus_sampler_get_history_count()`, `argus_sampler_has_pending()`) for real-time inspection of sequence penalty buffers and uncommitted sample states.

`libargus` is an ultra-lean, high-performance, model-agnostic inference wrapper engineered to consolidate LLM text generation, Whisper-based speech-to-text (ASR), Speech-LLM text-to-speech (TTS), and **bleeding-edge Multimodal (Vision, Audio, and Video) encoding and evaluation** pipelines into a single process-global native execution runtime.

Built directly on top of the modular **GGML** and **llama.cpp (libmtmd)** compute engines, `libargus` provides a unified, thread-safe C API designed explicitly for frictionless, zero-copy compilation alongside modern unmanaged orchestration frameworks, featuring out-of-the-box structural alignment for the JDK 22+ **Project Panama Foreign Function & Memory (FFM) API**.

---

## Installation & Dependency Setup

`libargus` is distributed on Maven Central for JDK 22+ (Project Panama FFM).

### Maven (`pom.xml`)
```xml
<dependencies>
    <!-- Core Java Panama FFM Bindings & High-Level API -->
    <dependency>
        <groupId>cc.projectargus</groupId>
        <artifactId>libargus-core</artifactId>
        <version>1.6.4</version>
    </dependency>

    <!-- Optional: Platform Native Runtime Provider (Automatic SPI Extraction) -->
    <dependency>
        <groupId>cc.projectargus</groupId>
        <artifactId>libargus-native-linux-cpu</artifactId>
        <version>1.6.4</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

### Gradle (`build.gradle.kts`)
```kotlin
dependencies {
    // Core Java Panama FFM Bindings & High-Level API
    implementation("cc.projectargus:libargus-core:1.6.4")

    // Optional: Platform Native Runtime Provider (Automatic SPI Extraction)
    runtimeOnly("cc.projectargus:libargus-native-linux-cpu:1.6.4")
}
```

> [!TIP]
> **JVM Runtime Requirement:** Because `libargus` leverages Project Panama Foreign Function & Memory (FFM) downcalls, you must pass `--enable-native-access=ALL-UNNAMED` to your JVM execution arguments.
> 
> **Platform Native Runtimes:** Precompiled SPI runtime artifacts are published to Maven Central for:
> * `libargus-native-linux-cpu`
> * `libargus-native-linux-cuda`
> * `libargus-native-linux-rocm`
> * `libargus-native-linux-vulkan`
> * `libargus-native-windows-cpu`
> * `libargus-native-windows-cuda`
> * `libargus-native-windows-vulkan`
> * `libargus-native-macos-metal`
> 
> If compiling the native C++ library from source, you only need `libargus-core` and can pass `-Dcc.projectargus.libargus.path=/path/to/libargus.so` or place the binary on `java.library.path`.

---

## Core Architectural Pillars

*   **Process-Global Backend Singularity:** Eliminates VRAM fragmentation and multi-context driver race conditions by orchestrating a singular, shared initialization pathway (`ggml_backend_load_all()`) across text, audio, speech, and multimodal subsystems.
*   **Decoupled Weights & Execution:** Separates model weight loading (`argus_model_t`) from evaluation context memory states (`argus_context_t`), allowing model reuse across multiple concurrent sessions.
*   **Bleeding-Edge Multimodal Projectors:** Integrates the new `libmtmd` C++ engine to ingest raw bitmaps, audio PCM arrays, and video files/streams. Tokenizes prompts and media into a unified chunk sequence, executes projection on the GPU, and automatically configures M-RoPE position grids and non-causal attention matrices.
*   **Unmanaged Video Iteration Pipe:** Decodes and streams video files frame-by-frame using internal FFmpeg subprocess pipes, yielding raw RGB frames or localized timestamp text chunks (e.g., `[12m34s]`) at a specified target frame rate.
*   **Pointers-Only FFM Alignment:** Replaces pass-by-value and volatile C++ polymorphic boundaries with strictly aligned, flat C functions accepting pointers. Structure padding is manually packed to prevent compilers from injecting alignment gaps (exact 56-byte `argus_sampler_params_t`).
*   **Absolute Zero-Copy Memory Boundaries:** Eliminates JVM heap primitive arrays (`int[]`, `float[]`) across hot paths. Integrates Project Panama `MemorySegment` parameters directly, allowing token tapes, audio waves, and video frames to generate speech and text with zero GC footprint.
*   **Selective Concurrency Locking:** Integrates context-level mutex synchronization to allow thread-safe decoding and context operations while enabling fully lock-free, concurrent tokenizer accesses on read-only models.
*   **Zero-Allocation Persistent Sampler & Sequence Isolation:** Caches unmanaged sampler chains per sequence slot, preserving token history sequences for repetition penalties and DRY n-gram suppression across decoding passes without per-token heap allocation overhead.
*   **Deterministic Stochastic Seeding & RNG Continuity:** Exposes 32-bit RNG seeds with seamless state preservation across parameter and logit bias mutations alongside temperature, top-p, min-p, top-k, repetition, frequency, presence, and DRY penalty hyperparameter envelopes.
*   **Coordinate-Decoupled Priming & Lifecycle:** Supports explicit priming (`primeSampler`) of penalty histories with tagged coordinate tracking to guide generation without polluting KV caches, alongside instant slot resets (`resetSampler`) and rollback replays (`truncateSampler`).
*   **Speculative & MTP Acceleration:** Incorporates native verification loops for traditional speculative drafting and Multi-Token Prediction (`draft-mtp`) directly inside the C++ execution layer with lockstep KV cache synchronization.
*   **Dynamic Sequence Slot Sizing & Unified KV Sharing:** Automatically allocates 100% of context memory to single-sequence generation (`seq_max = 1`) while supporting dynamic cross-sequence KV cell sharing (`kv_unified = true`) across speculative drafting and MTP tracks.
*   **KV Cache Quantization:** Supports native configurations (`type_k` and `type_v` cache enums) to offload memory footprints to Q8_0, Q4_0, or other optimized formats.
*   **Zero-Allocation Vocab & GGUF Metadata Introspection:** Exposes safe, unmanaged boundaries to lookup special vocab tokens (BOS, EOS, EOT, PAD), verify End-Of-Generation (EOG) conditions, and dynamically enumerate GGUF dictionary entries.
*   **Native VRAM Budgeting & Structural Introspection:** Exposes safe, unmanaged C & Project Panama FFM functions (`argus_model_kv_bytes_per_token`, `argus_model_estimate_vram_bytes`, `argus_model_size`) to calculate dynamic per-token KV footprints and total VRAM requirements without FFI allocation overhead.
*   **Dynamic Context CPU Thread Scaling:** Exposes thread-safe C & Project Panama FFM APIs (`argus_set_n_threads`, `argus_get_n_threads`, `argus_get_n_threads_batch`, `argus_audio_set_n_threads`) allowing CPU power governors to dynamically tune single-token decoding and batch prefilling thread allocations on live contexts without tearing down contexts or purging KV state.
*   **M-RoPE & Multidimensional Rollback Synchronization:** Native detection and position tracking for Multimodal Rotary Position Embeddings (M-RoPE / IM-RoPE). Automatically handles multidimensional temporal/spatial position vectors with zero-allocation introspection (`nPosPerEmbd()`, `isMRoPE()`).
*   **Automagic KV Cache Truncation & Prefix Rollback:** Automatically prunes invalidated KV cache cells on prefix reuse when `start_pos <= seq_pos_max`, establishing strict sequence monotonicity across 1D-RoPE and M-RoPE architectures with synchronized speculative draft context clearing.

---

## Codebase Topology

```
libargus/
├── CMakeLists.txt         # Layer-0 dependency isolation & optimization matrix
├── include/
│   └── libargus.h         # Master C ABI stable layout definitions
├── src/
│   ├── argus_internal.h   # Shared private structures (model & context)
│   ├── argus_common.cc    # Global backend lifecycles & hardware registries
│   ├── argus_text.cc      # Llama model/context handling, speculative loops & TTS
│   ├── argus_audio.cc     # Whisper model contexts & transcription (ASR)
│   └── argus_multimodal.cc# Multimodal context, media loaders, video pipes, and evaluation
└── bindings/java/         # Idiomatic Project Panama FFM binding module
    └── src/main/java/cc/projectargus/libargus/
        ├── ArgusBackend.java          # Global device telemetry & backend initialization
        ├── ArgusModel.java            # Unmanaged GGUF weights manager (AutoCloseable)
        ├── ArgusContext.java          # Core text evaluation context session
        ├── ArgusContextConfig.java    # Text context generation parameters
        ├── ArgusSamplerConfig.java    # Extended sampling configuration parameters
        ├── ArgusAudioContext.java     # Whisper speech-to-text transcription engine
        ├── ArgusMultimodalContext.java# Loaded multimodal projector context (AutoCloseable)
        ├── ArgusBitmap.java           # Raw/parsed RGB pixel or PCM audio sample buffer
        ├── ArgusVideo.java            # Frame iterator for video files or buffer pipes
        ├── ArgusVideoItem.java        # Reusable frame/timestamp container for video processing
        ├── ArgusInputChunks.java      # Tokenized multimodal prompt chunks container
        └── internal/
            ├── ArgusLayouts.java      # Panama C-to-Java struct layout definitions
            └── ArgusBindings.java     # Dynamic shared library method handle loader
```

---

## Build & Dependency Architecture

`libargus` enforces a pristine source configuration. It discards bloated upstream server implementations, legacy CLI targets, and unneeded dependencies by utilizing CMake-level build-layer vendoring (`FetchContent`). Upstream components are downloaded, configured, and statically linked inside the unmanaged compilation pass.

### Compilation Matrices

To compile the highly optimized shared binary target (`libargus.so` or `argus.dll`), execute the target generation commands from the root directory:

```bash
# Generate the optimized unmanaged compute graph project (enabling CUDA acceleration)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

# Compile the final unified system binary
cmake --build build --config Release -j $(nproc)
```

---

## Quickstart: Idiomatic Java Developer Experience

`libargus` shields JVM developers from complex pointer arithmetic, structure alignment gaps, and manual attention mask scheduling. Below is the high-level, memory-safe, and auto-closeable Java API usage pattern.

### Text & Audio Transcription (ASR)
```java
import cc.projectargus.libargus.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class Main {
    public static void main(String[] args) {
        // Initialize global backends (CUDA/CPU)
        ArgusBackend.init();

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, Path.of("models/llama-3-8b.gguf"), 99, true)) {
             
             // Initialize context configurations
             ArgusContextConfig config = new ArgusContextConfig.Builder(4096)
                 .cpuThreads(8)
                 .typeK(ArgusContextConfig.KV_TYPE_Q4_0) // Quantize KV Cache
                 .typeV(ArgusContextConfig.KV_TYPE_Q4_0)
                 .build();
                 
             try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                 // Run text evaluation & generation loop...
             }
        } finally {
             // Tear down native backend drivers
             ArgusBackend.free();
        }
    }
}
```

### Bleeding-Edge Multimodal Prompting (Vision/Video/Audio)
Using a vision-capable GGUF model along with its multimodal projector (`mmproj`):
```java
import cc.projectargus.libargus.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;

public class MultimodalApp {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (Arena arena = Arena.ofConfined();
              ArgusModel baseModel = ArgusModel.load(arena, Path.of("models/qwen2-vl-7b-it.gguf"), 99, true);
              ArgusContext context = ArgusContext.init(arena, baseModel, new ArgusContextConfig.Builder(8192).build());
              // Load the multimodal adapter context
              ArgusMultimodalContext mctx = ArgusMultimodalContext.init(arena, baseModel, Path.of("models/qwen2-vl-7b-it.mmproj"), 4, true)) {

            // 1. Load an image or audio file into unmanaged memory
            try (ArgusBitmap image = ArgusBitmap.loadFile(arena, mctx, Path.of("media/cat.png"), false)) {
                
                // 2. Tokenize prompt text replacing the media marker
                String prompt = "<__media__>\nDescribe what you see in this image.";
                try (ArgusInputChunks chunks = mctx.tokenize(arena, prompt, true, List.of(image))) {
                    
                    // 3. Evaluate chunks (handles image projection on GPU & M-RoPE position grids)
                    int newNPast = context.evalMultimodalChunks(mctx, chunks, 0, 0, 1024, true);
                    System.out.println("Prompt evaluated. Ready to sample output tokens! New position: " + newNPast);
                }
            }
        } finally {
            ArgusBackend.free();
        }
    }
}
```

### Frame-by-Frame Video Stream Processing
```java
// Iterate and read frames/timestamps from a video file sequentially
try (ArgusVideo video = ArgusVideo.loadFile(arena, mctx, Path.of("media/video.mp4"), 4.0f, 5000);
     ArgusVideoItem item = new ArgusVideoItem()) {
    while (video.readNext(item)) {
        if (item.bitmap() != null) {
            // Process the extracted RGB frame bitmap (ownership is managed by ArgusVideoItem)
            ArgusBitmap frame = item.bitmap();
            // ...
        } else if (item.text() != null) {
            // Received a video timestamp chunk (e.g. "[00m05s]")
            System.out.println("At timestamp: " + item.text());
        }
    }
}
```

### Extracting Semantic Text Embeddings
Retrieve float embedding vectors from dedicated models (e.g., `jina-embeddings-v3`):
```java
import cc.projectargus.libargus.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

public class EmbeddingsApp {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, Path.of("models/jina-embeddings-v3-Q4_K_M.gguf"), 99, true)) {

            // Initialize context configuration with embeddings enabled
            ArgusContextConfig config = new ArgusContextConfig.Builder(512)
                .cpuThreads(4)
                .embeddings(true)
                .build();

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                // Tokenize and evaluate prompt
                String text = "text-matching: Retrieve semantic vector for this sentence.";
                MemorySegment textSeg = arena.allocateFrom(text);
                
                MemorySegment tokenBuf = arena.allocate(ValueLayout.JAVA_INT, 512);
                int nTokens = context.tokenize(textSeg, tokenBuf, true);
                
                context.decodeBatch(tokenBuf, nTokens, 0, 0, false);

                // Retrieve embedding vector (e.g., 1024 dimensions)
                int expectedDim = 1024;
                MemorySegment embeddingsBuf = arena.allocate(ValueLayout.JAVA_FLOAT, expectedDim);
                int nFloats = context.getEmbeddings(0, embeddingsBuf, expectedDim);
                
                System.out.println("Retrieved embeddings containing " + nFloats + " floats.");
            }
        } finally {
            ArgusBackend.free();
        }
    }
}
```

### Model Metadata & Vocabulary Introspection

Query special tokens and traverse GGUF model configurations directly from unmanaged memory:

```java
// Access model vocabulary metadata
int bos = model.vocabBos();
int eos = model.vocabEos();
int eot = model.vocabEot(); // End-Of-Turn token for chat models
int nTokens = model.vocabNTokens(); // Vocabulary capacity
boolean isEog = model.vocabIsEog(sampledToken); // Native End-Of-Generation verification

// Query model architecture dimensions & parameters
int nEmbd = model.nEmbd(); // Embedding dimension size (e.g. 1024)
int nCtxTrain = model.nCtxTrain(); // Context length training ceiling
int nLayer = model.nLayer(); // Transformer layers count
int nHead = model.nHead(); // Attention head count
long nParams = model.nParams(); // Total model parameters count
long modelSize = model.modelSize(); // Total model weight footprint in VRAM (bytes)
String desc = model.desc(); // Human-readable architecture string (e.g. "llama 8B Q4_K_M")

// Pre-allocation VRAM budgeting calculations
long kvPerToken = model.kvBytesPerToken(ArgusContextConfig.KV_TYPE_Q4_0, ArgusContextConfig.KV_TYPE_Q4_0); // KV bytes / token
long estVram = model.estimateVramBytes(65536, ArgusContextConfig.KV_TYPE_Q4_0, ArgusContextConfig.KV_TYPE_Q4_0); // Total VRAM for 64k tokens

// Inspect GGML quantization block & element sizing primitives
long elemSize = ArgusModel.quantTypeSize(ArgusContextConfig.KV_TYPE_Q4_0); // Block byte size
int blockSize = ArgusModel.quantBlockSize(ArgusContextConfig.KV_TYPE_Q4_0); // Element count per block

// Retrieve metadata strings by key name
String modelArch = model.getMetadataValue("general.architecture"); // e.g. "qwen2vl"
String modelName = model.getMetadataValue("general.name"); // e.g. "Qwen2 VL 2B Instruct"

// Traverse and inspect the complete metadata dictionary
java.util.Map<String, String> metadata = model.getMetadataMap();
metadata.forEach((key, val) -> System.out.println(key + " -> " + val));
```

### Cancelable Native Prefill Batching

Execute long sequence prefilling with automatic native auto-chunking (preventing context batch overflows) and clean, cross-thread Java cancellation:

```java
try (ArgusContext context = ArgusContext.init(arena, model, config);
     ArgusAbortFlag abortFlag = new ArgusAbortFlag()) {

    // 1. Submit a large tokenized prompt. If it exceeds n_batch, libargus chunks it natively.
    // Pass the abortFlag directly to allow safe cancellation from other threads.
    int res = context.decodeBatch(tokensSeg, nTokens, 0, 0, true, abortFlag);
    if (res == -2) {
        System.out.println("Prefill decoding was cancelled early!");
    }

    // 2. In your UI or coroutine cancel handler thread, simply call:
    // abortFlag.abort();
}
```

### Extended Autoregressive Sampling & Persistent Sampler Caching

Configure rich sampling profiles (`top_p`, `min_p`, `top_k`, `temperature`, `seed`, repetition penalties, and DRY n-gram penalties) with zero per-token heap allocation:

```java
// Create custom sampling configuration profile with deterministic RNG seed
ArgusSamplerConfig samplerConfig = new ArgusSamplerConfig.Builder()
    .temperature(0.7f)
    .seed(42) // 32-bit deterministic RNG seed (-1 for random entropy)
    .topP(0.90f)
    .minP(0.05f)
    .topK(40)
    .repeatPenalty(1.1f)
    .repeatLastN(64)
    .frequencyPenalty(0.0f)
    .presencePenalty(0.0f)
    .dry(0.8f, 1.75f, 2, -1) // DRY (Don't Repeat Yourself) n-gram suppression
    .build();

while (generating) {
    // 1. Evaluate batch on sequence slot 0 (requesting terminal logits)
    context.decodeBatch(batch);

    // 2. Persistent sampler caches unmanaged chain and updates token history in place
    int token = context.sampleToken(0, samplerConfig);
    if (model.vocabIsEog(token)) break;
}
```

### Sampler Lifecycle & Decoupled Prompt Priming

Explicitly manage sequence slot sampler penalty histories without polluting KV cache states:

```java
// 1. Prime penalty history with initial system prompt tokens (bypassing KV cache pollution)
int[] primeTokens = new int[] { 100, 101, 102 };
MemorySegment primeSeg = arena.allocate(ValueLayout.JAVA_INT, primeTokens.length);
for (int i = 0; i < primeTokens.length; i++) primeSeg.setAtIndex(ValueLayout.JAVA_INT, i, primeTokens[i]);
context.primeSampler(0, primeSeg, primeTokens.length);

// 2. Roll back sampler history to prefix length during branch pruning (replays surviving tokens)
context.truncateSampler(0, 64);

// 3. Inspect retained token count and pending sample status
int historyCount = context.getSamplerHistoryCount(0); // e.g. 64 tokens
boolean hasPending = context.hasSamplerPending(0);    // false when committed

// 4. Reset sequence slot sampler chain and history to pristine state
context.resetSampler(0); // Pass -1 to reset all sequence slots
```

### Model-Agnostic Logit Bias Sampling

Enforce strict zero-allocation logit steering (e.g. banning reasoning tokens or boosting specific completions) by allocating bias segments once at the start of a generation session and reusing them across hot-path sampling steps:

```java
try (Arena sessionArena = Arena.ofConfined()) {
    // Define biased tokens and their steering weights (e.g. -Float.MAX_VALUE to ban)
    int[] steerTokens = new int[] { 151644, 151645 }; // <__think__> tags
    float[] steerValues = new float[] { -Float.MAX_VALUE, -Float.MAX_VALUE };

    // Allocate unmanaged struct segment ONCE outside the hot generation loop
    MemorySegment biasSeg = sessionArena.allocate(ArgusLayouts.LOGIT_BIAS, steerTokens.length);
    for (int i = 0; i < steerTokens.length; i++) {
        biasSeg.setAtIndex(ValueLayout.JAVA_INT, i * 2, steerTokens[i]);
        biasSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2 + 1, steerValues[i]);
    }

    while (generating) {
        context.decodeBatch(batch);
        
        // Zero-copy, zero-allocation token generation downcall passing raw pointer with extended sampler config
        int token = context.sampleTokenWithBias(
            0, samplerConfig, biasSeg, steerTokens.length
        );
        if (model.vocabIsEog(token)) break;
    }
}
```

---

## Sampler State Machine & Invariants

`libargus` enforces strict state-machine invariants across unmanaged memory boundaries to guarantee thread safety, mathematical correctness, and reproducible sampling:

| Subsystem / Contract | Invariant / Behavior | Native C ABI / Panama FFM |
|---|---|---|
| **Continuation Commit** | Pending sample state machine. Normal continuation commits tokens directly to history without chain reset, replay, or RNG interruption. | `argus_decode_batch()`, `argus_sampler_has_pending()` |
| **Logits Ownership** | Single-consumption model. Returns `-2` if target sequence was not evaluated last, if logits were already sampled, or if state was mutated. | `argus_sample_token_ext()` returns `-2` |
| **Coordinate Decoupling** | History entries are tagged with `kv_pos`. Primed prompt tokens (`kv_pos = -1`) survive KV rollbacks; only orphaned branch tokens (`kv_pos >= start_pos`) are pruned. | `argus_sampler_prime()`, `argus_decode_batch()` |
| **Entropy & RNG Seeding** | 32-bit seed (`0xFFFFFFFF` / `-1` for random entropy). Pure greedy argmax when `temperature <= 0.0f`. Distribution sampling when `temperature > 0.0f`. | `argus_sampler_params_t.seed` |
| **Monotonic RNG Continuity** | Dynamic hyperparameter tuning, partial KV rollbacks, and truncations clone the active distribution sampler RNG state without restarting to draw #0. | `llama_sampler_clone()` on reconfigure/rollback |
| **Multimodal Synchronization**| Per-context mutex serialization across chunk evaluation, eliminating races against concurrent text decode, sampling, and KV mutation. | `argus_eval_multimodal_chunks()` holds `ctx->mtx` |
| **Penalty History Replay** | Rebuilding filter chains on reconfiguration or rollback zero-allocation replays surviving sequence tokens via `llama_sampler_accept()`. | Persistent slot history buffer |
| **Stale Logits Invalidation** | Any KV cache mutation (`clearCacheSlot`), decode failure, multimodal projection error, or slot reset immediately invalidates pending logits. | `invalidate_seq_logits()` |
| **History Introspection** | Zero-allocation real-time queries for retained token count (primed + committed) and uncommitted sample status. | `argus_sampler_get_history_count()`, `argus_sampler_has_pending()` |

---

## Native Memory Layout & Project Panama Alignment

All native C structures are packed with explicit padding to guarantee exact 8-byte alignment across x86-64 and AArch64 without compiler layout drift:

### `argus_sampler_params_t` (56 Bytes, 8-Byte Aligned)
```
Offset  Size  Type       Field Name            Description
---------------------------------------------------------------------------------------------
0       4     float      temperature           Entropy control (<= 0.0f is greedy / argmax)
4       4     float      repeat_penalty        Repetition suppression multiplier
8       4     int32_t    repeat_last_n         Lookback token window (0 defaults to 64)
12      4     float      frequency_penalty     Frequency penalty factor (0.0f is disabled)
16      4     float      presence_penalty      Presence penalty factor (0.0f is disabled)
20      4     float      top_p                 Nucleus sampling threshold (>= 1.0f is disabled)
24      4     float      min_p                 Min probability threshold (<= 0.0f is disabled)
28      4     int32_t    top_k                 Top-K candidate cap (<= 0 is disabled)
32      4     float      dry_multiplier        DRY penalty multiplier (0.0f is disabled)
36      4     float      dry_base              DRY exponential base (defaults to 1.75f)
40      4     int32_t    dry_allowed_length    DRY allowed n-gram length (defaults to 2)
44      4     int32_t    dry_penalty_last_n    DRY lookback window (-1 matches full context)
48      4     uint32_t   seed                  RNG seed (0xFFFFFFFF = random / LLAMA_DEFAULT_SEED)
52      4     uint8_t[4] reserved_padding      Explicit alignment padding securing 8-byte boundary
---------------------------------------------------------------------------------------------
Total Struct Byte Size: 56 bytes (0 padding holes)
```

### Dynamic CPU Thread Allocation & Governor Control

Dynamically adjust single-token decoding (`n_threads`) and batch prefilling (`n_threads_batch`) allocations on an existing context session without purging KV state or recreating contexts:

```java
// Query active thread counts from the native context
int curGenThreads = context.getNThreads();
int curBatchThreads = context.getNThreadsBatch();

// Dynamically tune CPU thread allocation (e.g. scaling down during thermal throttling or background tasks)
context.setNThreads(4, 8); // 4 threads for generation, 8 threads for prompt batch prefilling

// Audio contexts (Whisper ASR) also support dynamic acoustic thread scaling
audioContext.setNThreads(4);
```

### M-RoPE Introspection & Automagic KV Cache Rollback

Query multidimensional rotary position topologies and manage KV cache sequence rollbacks for agentic ReAct loops and Longest Common Prefix (LCP) prompt reuse:

```java
// 1. Inspect M-RoPE architecture properties on loaded models
boolean isMRoPE = model.isMRoPE();         // true for Qwen2-VL, Qwen2.5-VL, etc.
int nPosPerEmbd = model.nPosPerEmbd();     // 4 for M-RoPE, 1 for standard 1D-RoPE

// 2. Query active sequence position boundaries directly from unmanaged memory
int posMax = context.getSeqPosMax(0);      // High-water mark position (or -1 if empty)
int posMin = context.getSeqPosMin(0);      // Low-water mark position (or -1 if empty)

// 3. Explicit KV Cache Tail Truncation: Roll back sequence slot to position 128
context.clearCacheSlot(0, 128, -1);        // Synchronously clears primary & speculative draft caches

// 4. Automagic Prefix Rollback: Submitting a batch at start_pos <= seq_pos_max
// automatically prunes [start_pos, -1) to guarantee monotonicity without manual clearing:
int res = context.decodeBatch(newBranchTokens, branchLength, 128, 0, true);
```

---

## Verification & Testing Suite

Validate unmanaged tensor boundary compliance and multi-model processing thread re-entrancy by running the native and Java integration testing pipelines:

```bash
# Run native C unit assertions
./build/bin/test_libargus

# Run JUnit / Panama FFM integration tests
./gradlew test
```

---

## Engineering Methodology & Development Velocity

`libargus` was architected, engineered, and brought to stable release in a single continuous sprint. To achieve this velocity without compromising performance or memory safety, a distinct division of execution was enforced:

* **Human Core (Architecture & Systems Design):** Every critical memory semantic, low-level constraint, and hardware optimization boundary was explicitly designed and driven by human engineering. This includes off-heap Arena lifecycle boundaries (`Arena.ofConfined`), strict 1:1 manual struct alignment packing to prevent cross-compiler layout drift, mutable off-heap asset recycling paths (`ArgusVideoItem`) to bypass JVM GC overhead, and the $O(1)$ zero-copy interleaved logit steering matrix (`argus_logit_bias_t`).
* **AI Core (Boilerplate Compilation Pass):** Large Language Models were leveraged strictly as high-speed syntactic compilers. AI was used to rapidly generate repetitive unmanaged C-to-Java downcall bindings, parameter builder boilerplate, and tedious structural Java mapping layout strings based directly on explicit engineering blueprints.

This hybrid methodology treats AI not as an unguided code generator, but as an advanced text compiler—accelerating the delivery of zero-allocation, mechanically sympathetic systems code while ensuring total architectural control remains human-driven.

---

## Upstream Integration & Project Roadmap

`libargus` is engineered strictly as **Layer 0 (The Core Execution Bedrock)** for low-latency, performance-critical JVM platforms. It provides the raw compute foundation required for zero-allocation native tensor orchestration via Project Panama.

This engine serves as the high-throughput infrastructure for a broader cognitive platform. To view the high-level roadmap detailing how this runtime block interfaces with the upcoming Layer 1 stateful cognitive core (L-TABB) and the unified system dashboard, visit the master project organization landing page at [ProjectArgus.cc](https://github.com/ProjectArgus-cc).

---

## Licensing & Attribution

`libargus` is released open-source under the MIT License. This software integrates and links against computational tensor primitives derived from `llama.cpp` (including `libmtmd`) and `whisper.cpp`.
