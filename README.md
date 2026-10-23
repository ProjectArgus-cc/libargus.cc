# libargus
## An unmanaged native AI execution runtime engineered for zero-allocation steady-state inference, consolidating Vision, Speech, and LLM compute pipelines behind unified Project Panama FFM (JDK 22+) and JNI (JDK 17+ / Android) boundaries.

[![Release Pipeline](https://github.com/ProjectArgus-cc/libargus.cc/actions/workflows/release.yml/badge.svg)](https://github.com/ProjectArgus-cc/libargus.cc/actions/workflows/release.yml)
[![Maven Central](https://img.shields.io/maven-central/v/cc.projectargus/libargus-core.svg?label=Maven%20Central&color=blue)](https://central.sonatype.com/artifact/cc.projectargus/libargus-core)
[![JDK Target](https://img.shields.io/badge/JDK-22%2B%20Panama%20FFM%20%7C%2017%2B%20JNI-orange.svg)](https://openjdk.org/jeps/454)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`libargus` is an ultra-lean, model-agnostic inference runtime that consolidates **LLM text generation**, **semantic text embeddings**, **Whisper speech-to-text (ASR)**, **Speech-LLM text-to-speech (TTS)**, and **Multimodal (Vision, Audio, and Video)** pipelines into a single process-global native shared library (`libargus.so` / `argus.dll` / `libargus.dylib`).

Built directly on top of **GGML**, **llama.cpp (`libmtmd`)**, and **whisper.cpp**, `libargus` provides a strict, thread-safe C ABI (`include/libargus.h`) and two 100% feature-symmetric, package-compatible (`cc.projectargus.libargus.*`) Java bindings:

| Binding Artifact | Runtime Target | FFI Mechanism | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`cc.projectargus:libargus-core`** | **JDK 22+** (`--release 22`) | Project Panama FFM (`MemorySegment`, `Arena`, `Linker`) | Ultra-low-latency server & desktop JVMs (`--enable-native-access=ALL-UNNAMED`) |
| **`cc.projectargus:libargus-jni`** | **JDK 17+ & Android** (`--release 17`) | Unified JNI Bridge (`DirectByteBuffer` & primitive arrays) | Android apps/NDK, enterprise JDK 17/21 services, zero JVM flag requirements |

---

## Installation & Dependency Setup

Choose **either** `libargus-core` (JDK 22+ Panama FFM) **or** `libargus-jni` (JDK 17+ / Android), plus the optional precompiled native SPI runtime for your target hardware:

### Gradle (`build.gradle.kts`)
```kotlin
dependencies {
    // Option A: JDK 22+ Project Panama FFM Bindings
    implementation("cc.projectargus:libargus-core:2.3.1")

    // Option B: JDK 17+ / Android Drop-In JNI Bindings (100% API Parity)
    // implementation("cc.projectargus:libargus-jni:2.3.1")

    // Optional: LangChain4j Integration (works seamlessly with either libargus-core or libargus-jni)
    // implementation("cc.projectargus:libargus-langchain4j:2.3.1")

    // Optional: Precompiled Native SPI Provider (auto-extracts on both FFM & JNI)
    runtimeOnly("cc.projectargus:libargus-native-linux-cpu:2.3.1")
}
```

### Maven (`pom.xml`)
```xml
<dependencies>
    <!-- Option A: JDK 22+ Project Panama FFM Bindings (or use libargus-jni for JDK 17+ / Android) -->
    <dependency>
        <groupId>cc.projectargus</groupId>
        <artifactId>libargus-core</artifactId>
        <version>2.3.1</version>
    </dependency>

    <!-- Optional: Precompiled Native SPI Provider -->
    <dependency>
        <groupId>cc.projectargus</groupId>
        <artifactId>libargus-native-linux-cpu</artifactId>
        <version>2.3.1</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

> [!TIP]
> **Precompiled Native SPI Classifiers (`--release 17`):**
> `libargus-native-linux-cpu` · `libargus-native-linux-cuda` · `libargus-native-linux-rocm` · `libargus-native-linux-vulkan` · `libargus-native-windows-cpu` · `libargus-native-windows-cuda` · `libargus-native-windows-vulkan` · `libargus-native-macos-metal`
>
> When running with `libargus-core` (Project Panama FFM), pass `--enable-native-access=ALL-UNNAMED` to the JVM. `libargus-jni` requires no JVM flags. For custom-built binaries, pass `-Dcc.projectargus.libargus.path=/path/to/libargus.so` or place the library on `java.library.path` / Android `jniLibs/`.

---

## 60-Second Quickstarts (100% Source-Compatible Across FFM & JNI)

All high-level examples below compile and run identically on both `libargus-core` and `libargus-jni` without manual `Arena` or pointer management.

### 1. Streaming LLM Text Generation
```java
import cc.projectargus.libargus.*;
import java.nio.file.Path;

public class QuickstartText {
    public static void main(String[] args) {
        ArgusBackend.init();

        try (ArgusModel model = ArgusModel.load(Path.of("models/llama-3-8b.gguf"), 99, true);
             ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(4096)
                 .cpuThreads(8)
                 .typeK(ArgusContextConfig.KV_TYPE_Q4_0)
                 .typeV(ArgusContextConfig.KV_TYPE_Q4_0)
                 .build())) {

            ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder()
                .temperature(0.7f)
                .topP(0.90f)
                .seed(42)
                .build();

            // Stateful UTF-8 streaming (never corrupts split multi-byte emojis or CJK tokens)
            String response = context.generate("Explain quantum entanglement:", 256, sampler, System.out::print);

            // Or stream with an early-stopping Predicate<String> via generateUntil:
            String firstLine = context.generateUntil("One-line summary of TCP/IP:", 128, sampler, piece -> {
                System.out.print(piece);
                return !piece.contains("\n"); // Return false to stop generation early
            });
        } finally {
            ArgusBackend.free();
        }
    }
}
```

### 2. Semantic Text Embeddings
```java
ArgusContextConfig embedConfig = new ArgusContextConfig.Builder(512)
    .embeddings(true)
    .causalAttn(false) // Bidirectional non-causal attention for BERT / Gemma2 / Jina models
    .build();

try (ArgusModel model = ArgusModel.load(Path.of("models/jina-embeddings-v3.gguf"), 99, true);
     ArgusContext context = ArgusContext.init(model, embedConfig)) {

    float[] embedding = context.extractEmbeddings("Retrieve semantic vector for this sentence.");
    System.out.println("Extracted " + embedding.length + "-d vector.");
}
```

### 3. Multimodal Vision & Audio Prompting
```java
try (ArgusModel model = ArgusModel.load(Path.of("models/qwen2-vl-7b-it.gguf"), 99, true);
     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(8192).build());
     ArgusMultimodalContext mctx = ArgusMultimodalContext.init(model, Path.of("models/qwen2-vl-7b-it.mmproj"), 4, true);
     ArgusBitmap image = ArgusBitmap.loadFile(mctx, Path.of("media/cat.png"), false);
     ArgusInputChunks chunks = mctx.tokenize("<__media__>\nDescribe this image.", true, java.util.List.of(image))) {

    int nPast = context.evalMultimodalChunks(mctx, chunks, 0, 0, 1024, true);
    int firstToken = context.sampleToken(0, new ArgusSamplerConfig.Builder().temperature(0.2f).build());
    System.out.println("First generated piece: " + context.tokenToPiece(firstToken));
}
```

### 4. Whisper Speech-to-Text (ASR)
```java
try (ArgusAudioContext whisper = ArgusAudioContext.init(Path.of("models/whisper-base.en.gguf"), 4, 99)) {
    float[] pcm16kHz = loadNormalized16kHzAudio();
    String transcript = whisper.transcribe(pcm16kHz, pcm16kHz.length, 4096);
    System.out.println(transcript);
}
```

---

## Documentation & Deep-Dive Guides

| Guide | What You Will Learn |
| :--- | :--- |
| **[Java Developer Cookbook](docs/quickstart-java.md)** | Full end-to-end examples: Streaming LLM, Cooperative Cancellation (`ArgusAbortFlag`), Low-Level Token Loops, Logit Bias, Embeddings, Multimodal Vision/Video (`ArgusVideo`), Whisper ASR, Speech Synthesis (TTS), Sequence Slot Forking (`forkSlot`), VRAM Budgeting, Logging, and Performance Telemetry (`ArgusPerfTimings`). |
| **[LangChain4j Developer Cookbook](docs/quickstart-langchain4j.md)** | Copy-pasteable recipes for `LibArgusChatModel`, `LibArgusStreamingChatModel` (with `ArgusAbortFlag`), `LibArgusEmbeddingModel` (`causalAttn(false)` RAG), Zero-Trust Multimodal Vision/Audio (`allowedMediaRoot`), and shared `ArgusModel` VRAM lifecycles. |
| **[Android & JDK 17+ JNI Guide](docs/quickstart-android-jni.md)** | Android APK `jniLibs` setup, **safe `ParcelFileDescriptor` (`pfd.getFd()`) model loading**, zero-copy `DirectByteBuffer` (`*Direct`) methods, dynamic big.LITTLE CPU thread scaling (`setNThreads`), and ProGuard / R8 consumer rules. |
| **[Executable Doctests & Synthetic Models](docs/executable-doctests-and-synthetic-models.md)** | How `libargus` compiles and executes every Markdown documentation snippet across both Panama FFM and JNI using pure-Java, on-the-fly synthesized neural fixtures (`tiny-whisper.bin`, `tiny-mmproj-omni.gguf`, `tiny-vocoder.gguf`, `tiny-causal-260.gguf`). |
| **[Architecture & Native ABI Spec](docs/architecture-and-abi.md)** | Process-Global Backend Singularity, Thread-Safe Handle Leasing (`AbstractNativeResource`), Panama FFM Shared Arenas, 56-byte C ABI struct layouts (`argus_sampler_params_t`, `argus_perf_timings_t`), M-RoPE & KV cache rollback/shift mechanics, Heap Allocation Profiling (`scripts/AllocationProbe.java`), and CMake compilation matrices. |
| **[Release Operations Runbook](docs/release.md)** | 8-target cross-platform validation matrix, 44-asset Maven Central & GitHub Release cryptographic provenance (`manifest.json`), classifier verification, and release recovery workflows. |
| **[AI Agent & Maintainer Contract](AGENTS.md)** | Codebase topology, module boundaries, non-negotiable concurrency & memory invariants, 22-row Sampler State Machine contract table, mutation test anchors, and isolated verification commands. |

---

## Building from Source & Running Tests

```bash
# 1. Compile Portable Native Library + Test Suite (add -DGGML_CUDA=ON, -DGGML_HIP=ON, -DGGML_VULKAN=ON, or -DGGML_METAL=ON for GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DARGUS_TESTING=ON -DARGUS_MEDIA_REQUIRED=ON
cmake --build build --config Release -j $(nproc)
ctest --test-dir build --output-on-failure

# 2. Run Java Unit, Panama FFM, JNI & Differential Equivalence Suites
./gradlew :libargus-common:test :libargus-core:test :libargus-jni:test \
  -PskipCMake=true -PnativeTesting=true \
  -PnativePath="$PWD/build/lib/libargus_test.so" -PtestJdk=22
```

---

## Upstream Integration & Licensing

`libargus` serves as **Layer 0 (The Core Execution Bedrock)** for the [ProjectArgus.cc](https://github.com/ProjectArgus-cc) cognitive platform. Released open-source under the [MIT License](LICENSE), integrating compute primitives from `llama.cpp` (`libmtmd`) and `whisper.cpp`.
