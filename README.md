# libargus
## An unmanaged, zero-allocation native AI execution runtime consolidating Vision, Speech, and LLM compute pipelines behind a single Project Panama FFM boundary.
> [!NOTE]
> **v1.0.0 Stable — Unified Hardware Orchestration & Zero-Allocation ABI**
>

`libargus` is an ultra-lean, high-performance, model-agnostic inference wrapper engineered to consolidate LLM text generation, Whisper-based speech-to-text (ASR), Speech-LLM text-to-speech (TTS), and **bleeding-edge Multimodal (Vision, Audio, and Video) encoding and evaluation** pipelines into a single process-global native execution runtime.

Built directly on top of the modular **GGML** and **llama.cpp (libmtmd)** compute engines, `libargus` provides a unified, thread-safe C API designed explicitly for frictionless, zero-copy compilation alongside modern unmanaged orchestration frameworks, featuring out-of-the-box structural alignment for the JDK 22+ **Project Panama Foreign Function & Memory (FFM) API**.

---

## Core Architectural Pillars

*   **Process-Global Backend Singularity:** Eliminates VRAM fragmentation and multi-context driver race conditions by orchestrating a singular, shared initialization pathway (`ggml_backend_load_all()`) across text, audio, speech, and multimodal subsystems.
*   **Decoupled Weights & Execution:** Separates model weight loading (`argus_model_t`) from evaluation context memory states (`argus_context_t`), allowing model reuse across multiple concurrent sessions.
*   **Bleeding-Edge Multimodal Projectors:** Integrates the new `libmtmd` C++ engine to ingest raw bitmaps, audio PCM arrays, and video files/streams. Tokenizes prompts and media into a unified chunk sequence, executes projection on the GPU, and automatically configures M-RoPE position grids and non-causal attention matrices.
*   **Unmanaged Video Iteration Pipe:** Decodes and streams video files frame-by-frame using internal FFmpeg subprocess pipes, yielding raw RGB frames or localized timestamp text chunks (e.g., `[12m34s]`) at a specified target frame rate.
*   **Pointers-Only FFM Alignment:** Replaces pass-by-value and volatile C++ polymorphic boundaries with strictly aligned, flat C functions accepting pointers. Structure padding is manually packed to prevent compilers from injecting alignment gaps.
*   **Absolute Zero-Copy Memory Boundaries:** Eliminates JVM heap primitive arrays (`int[]`, `float[]`) across hot paths. Integrates Project Panama `MemorySegment` parameters directly, allowing token tapes, audio waves, and video frames to generate speech and text with zero GC footprint.
*   **Selective Concurrency Locking:** Integrates context-level mutex synchronization to allow thread-safe decoding and context operations while enabling fully lock-free, concurrent tokenizer accesses on read-only models.
*   **Speculative & MTP Acceleration:** Incorporates native verification loops for traditional speculative drafting and Multi-Token Prediction (`draft-mtp`) directly inside the C++ execution layer.
*   **KV Cache Quantization:** Supports native configurations (`type_k` and `type_v` cache enums) to offload memory footprints to Q8_0, Q4_0, or other optimized formats.
*   **Zero-Allocation Vocab & GGUF Metadata Introspection:** Exposes safe, unmanaged boundaries to lookup special vocab tokens (BOS, EOS, EOT, PAD), verify End-Of-Generation (EOG) conditions, and dynamically enumerate GGUF dictionary entries.

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

// Retrieve metadata strings by key name
String modelArch = model.getMetadataValue("general.architecture"); // e.g. "qwen2vl"
String modelName = model.getMetadataValue("general.name"); // e.g. "Qwen2 VL 2B Instruct"

// Traverse and inspect the complete metadata dictionary
java.util.Map<String, String> metadata = model.getMetadataMap();
metadata.forEach((key, val) -> System.out.println(key + " -> " + val));
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
        
        // Zero-copy, zero-allocation token generation downcall passing raw pointer
        int token = context.sampleTokenWithBias(
            seqId, temperature, repeatPenalty, biasSeg, steerTokens.length
        );
        if (token == model.vocabEos()) break;
    }
}
```

---

## Verification & Testing Suite

Validate unmanaged tensor boundary compliance and multi-model processing thread re-entrancy by running the native and Java integration testing pipelines:

```bash
# Run native C unit assertions
./build/test_libargus

# Run JUnit / Panama FFM integration tests
cd bindings/java && gradle test
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

This engine serves as the high-throughput infrastructure for a broader cognitive platform. To view the high-level roadmap detailing how this runtime block interfaces with the upcoming Layer 1 stateful cognitive core (L-TABB) and the unified system dashboard, visit the master project organization landing page at [ProjectArgus.cc](https://github.com/projectargus-cc).

---

## Licensing & Attribution

`libargus` is released open-source under the MIT License. This software integrates and links against computational tensor primitives derived from `llama.cpp` (including `libmtmd`) and `whisper.cpp`.
