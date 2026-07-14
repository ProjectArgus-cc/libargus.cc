# libargus
## An unmanaged, zero-allocation native AI execution runtime consolidating Vision, Speech, and LLM compute pipelines behind a single Project Panama FFM boundary.
> [!IMPORTANT]
> **v0.1.0 Alpha — Architectural Proof-of-Concept & ABI Freeze**
>
> `libargus` is public to solicit adversarial peer review on its low-level systems architecture, unmanaged compute graph consolidation, and Project Panama Foreign Function & Memory (FFM) boundary alignment.
>
> **Current Architecture & Active Refactoring Targets:**
> * **Panama FFM ABI Stability:** The unmanaged C ABI, manually packed structure padding (`reserved_padding[3]`), and pointer-only downcall boundaries are frozen and stable for JDK 22/25 integration.
> * **TTS Vocoder Execution:** The speech synthesis pipeline currently initializes ephemeral vocoder contexts per execution; migration to persistent session state to eliminate thread pool churn and achieve true zero-allocation execution is under active development.
> * **ASR Buffer Assembly:** The Whisper acoustic transcription buffer is undergoing optimization from linear accumulation to zero-copy linear pointer tracking with strict UTF-8 continuation byte boundary enforcement.
> * **SIMD Compilation Target:** The CMake build matrix currently defaults to host-native SIMD instruction generation (`-march=native`).

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
    ├── src/main/java/cc/projectargus/libargus/
    │   ├── ArgusBackend.java          # Global device telemetry & backend initialization
    │   ├── ArgusModel.java            # Unmanaged GGUF weights manager (AutoCloseable)
    │   ├── ArgusContext.java          # Core text evaluation context session
    │   ├── ArgusContextConfig.java    # Text context generation parameters
    │   ├── ArgusAudioContext.java     # Whisper speech-to-text transcription engine
    │   ├── ArgusMultimodalContext.java# Loaded multimodal projector context (AutoCloseable)
    │   ├── ArgusBitmap.java           # Raw/parsed RGB pixel or PCM audio sample buffer
    │   ├── ArgusVideo.java            # Frame iterator for video files or buffer pipes
    │   ├── ArgusInputChunks.java      # Tokenized multimodal prompt chunks container
    │   └── internal/
    │       ├── ArgusLayouts.java      # Panama C-to-Java struct layout definitions
    │       └── ArgusBindings.java     # Dynamic shared library method handle loader
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
             ArgusContextConfig config = new ArgusContextConfig.Builder()
                 .contextLength(4096)
                 .cpuThreads(8)
                 .typeK(ArgusBackend.KV_TYPE_Q4_0) // Quantize KV Cache
                 .typeV(ArgusBackend.KV_TYPE_Q4_0)
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
             ArgusContext context = ArgusContext.init(arena, baseModel, new ArgusContextConfig.Builder().build());
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
try (ArgusVideo video = ArgusVideo.loadFile(arena, mctx, Path.of("media/video.mp4"), 4.0f, 5000)) {
    ArgusVideo.VideoItem item;
    while ((item = video.readNext()) != null) {
        if (item.bitmap() != null) {
            // Process the extracted RGB frame bitmap
            try (ArgusBitmap frame = item.bitmap()) {
                // Tokenize and evaluate frame...
            }
        } else if (item.text() != null) {
            // Received a video timestamp chunk (e.g. "[00m05s]")
            System.out.println("At timestamp: " + item.text());
        }
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

## Licensing & Attribution

`libargus` is released open-source under the MIT License. This software integrates and links against computational tensor primitives derived from `llama.cpp` (including `libmtmd`) and `whisper.cpp`.
