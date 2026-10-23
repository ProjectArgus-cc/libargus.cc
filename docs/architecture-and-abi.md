# Architecture & Native ABI Specification

`libargus` is engineered as **Layer 0 (The Core Execution Bedrock)** for low-latency, mechanically sympathetic JVM and native platforms. It consolidates LLM text generation, Whisper-based speech-to-text (ASR), Speech-LLM text-to-speech (TTS), and multimodal (Vision, Audio, and Video) pipelines behind a single thread-safe C ABI (`include/libargus.h`) and unified JNI surface (`src/jni/argus_jni.cc`).

---

## 1. Core Architectural Pillars

* **Process-Global Backend Singularity:** Eliminates VRAM fragmentation and multi-context driver race conditions by orchestrating a singular, shared initialization pathway (`ggml_backend_load_all()`) across text, audio, speech, and multimodal subsystems.
* **Unified Single-Binary C ABI & JNI Surface:** Compiles C ABI exports and `Java_cc_projectargus_libargus_jni_*` entrypoints into a single shared library (`libargus.so` / `argus.dll` / `libargus.dylib`), avoiding multi-library dynamic linker races on Android Bionic namespaces and desktop JVMs while sharing a unified `AbstractNativeResource` read/write lease hierarchy across `:libargus-core` and `:libargus-jni`.
* **Thread-Safe Native Handle Leasing:** Wraps unmanaged pointers in `AbstractNativeResource` (`ArgusNativeResource` for Panama `MemorySegment` and `ArgusJniResource` for `long` handles) backed by `ReentrantReadWriteLock`. Downcalls acquire shared read leases via `try (var lease = res.lease())` or `withHandle(Function)` during invocations while `close()` acquires an exclusive write lease to invalidate handles, deterministically preventing use-after-free, concurrent closure races, and accidental lease leaks.
* **Project Panama Critical Downcalls & Architectural Allowlist:** Strictly restricts `Linker.Option.critical(false)` via `CRITICAL_ALLOWLIST` to 4 pure, atomic, non-blocking C ABI leaf operations (`argus_build_features`, `argus_abort_flag_is_requested`, `argus_last_error_code`, `argus_clear_error`). All other operations use standard downcalls, eliminating JNI/Panama safepoint costs on hot leaves without risking JVM hangs or heap deadlocks.
* **Independent Context Policy & Enforceable Native Model Ownership:** Context memory states (`argus_context_t` and `argus_multimodal_context_t`) retain their underlying `argus_model_t` reference via `argus_model_retain` / `argus_model_release`. Java `ArgusContext` queries model handles via `argus_context_get_model` under the context read lease, allowing contexts to remain fully operational even if the initial Java `ArgusModel` wrapper is closed. Deferred backend teardown (`g_backend_active_resources`) ensures memory safety during out-of-order JVM shutdown.
* **Panama FFM Shared Arena Concurrency Safety:** Employs an internal, context-owned `Arena.ofShared()` protected by reentrant lifecycle locks. Multi-threaded worker pools can concurrently dispatch downcalls on shared contexts without tripping `WrongThreadException`.
* **C ABI Exception Containment & Structured Diagnostics:** Fallible native operations contain C++ exceptions and report errors through bounded thread-local diagnostics (`argus_last_error_code`, `argus_last_error_message_copy`), mapped cleanly into `ArgusNativeException` on the Java side.
* **Spatial Memory Boundary Enforcement:** Validates `MemorySegment`, `DirectByteBuffer`, and primitive array boundaries and offsets prior to native calls via `ArgusCommonValidation`, `ArgusValidation`, and `ArgusJniValidation`, rejecting invalid Java buffer extents before entering the compute engine.
* **Decoupled Weights & Execution:** Separates model weight loading (`argus_model_t`, via filesystem paths or POSIX/CRT file descriptors `argus_model_load_fd`) from evaluation context memory states (`argus_context_t`), allowing weight reuse across multiple concurrent sessions.
* **Persistent Sampler & Sequence Isolation:** Caches unmanaged sampler chains per sequence slot, preserving token history sequences for repetition penalties and DRY n-gram suppression across decoding passes while retaining reusable sampler RNG state across reconfigurations and partial rollbacks.
* **Speculative & MTP Acceleration:** Provides synchronized dual-context plumbing (primary and speculative draft KV synchronization, rollback, and thread scaling) alongside Multi-Token Prediction (`draft-mtp`) acceleration directly inside the C++ execution layer.
* **Diagnostic Logging & Zero-Overhead Redirection:** Silences upstream C++ `stderr` pollution by default (`ARGUS_LOG_WARN`) across `llama.cpp`, `ggml`, `whisper.cpp`, and `libmtmd`. Features lock-free relaxed atomic hot-path checks, thread-local continuation tracking (`ARGUS_LOG_CONT`), environment variable overrides (`LIBARGUS_LOG_LEVEL`), and foreign callback exception barriers with in-flight quiescence tracking (`argus_log_in_flight_count`).

---

## 2. Native Memory Layout & Project Panama Alignment

Public C structures retain natural alignment and explicit reserved padding bytes so C compilers and Project Panama `MemoryLayout` definitions agree with zero implicit padding holes. Native diagnostics (`ArgusBackend.getBuildInfo()`) expose compiler-derived sizes, alignments, and field offsets (`src/argus_abi.h`) and verify them against Java `ArgusLayouts.diagnosticAbi()`.

### `argus_sampler_params_t` (56 Bytes, 4-Byte Aligned)
```text
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
52      4     uint8_t[4] reserved_padding      Reserved bytes; natural struct alignment remains 4
---------------------------------------------------------------------------------------------
Total Struct Byte Size: 56 bytes (0 implicit padding holes)
```

### `argus_perf_timings_t` (56 Bytes, 8-Byte Aligned)
```text
Offset  Size  Type       Field Name            Description
---------------------------------------------------------------------------------------------
0       8     double     t_start_ms            Context execution start timestamp in ms
8       8     double     t_load_ms             Model load duration in ms
16      8     double     t_p_eval_ms           Prompt prefill processing duration in ms
24      8     double     t_eval_ms             Autoregressive generation duration in ms
32      4     int32_t    n_p_eval              Prompt tokens evaluated
36      4     int32_t    n_eval                Generation tokens evaluated
40      4     int32_t    n_reused              KV cache reused graph iterations count
44      4     int32_t    n_reallocs_same       KV cache reallocations within the same buffer
48      4     int32_t    n_reallocs_diff       KV cache reallocations in different buffers
52      4     uint8_t[4] reserved_padding      Reserved bytes; struct size 56, alignment 8
---------------------------------------------------------------------------------------------
Total Struct Byte Size: 56 bytes (0 implicit padding holes)
```

### Other Public ABI Structs (`include/libargus.h`)
* **`argus_model_params_t` (16 Bytes, 8-Byte Aligned):** `model_path` (offset 0, 8B), `gpu_layers` (offset 8, 4B), `use_mlock` (offset 12, 1B), `reserved_padding[3]` (offset 13, 3B).
* **`argus_context_params_t` (40 Bytes, 8-Byte Aligned):** `draft_model` (offset 0, 8B), `context_length` (offset 8, 4B), `cpu_threads` (offset 12, 4B), `type_k` (offset 16, 4B), `type_v` (offset 20, 4B), `spec_draft_n_max` (offset 24, 4B), `u_batch` (offset 28, 4B), `n_seq_max` (offset 32, 4B), `enable_draft_mtp` (offset 36, 1B), `embeddings` (offset 37, 1B), `kv_unified` (offset 38, 1B), `reserved_padding[1]` (offset 39, 1B).
* **`argus_audio_params_t` (16 Bytes, 8-Byte Aligned):** `whisper_model_path` (offset 0, 8B), `cpu_threads` (offset 8, 4B), `gpu_layers` (offset 12, 4B).
* **`argus_token_batch_t` (32 Bytes, 8-Byte Aligned):** `tokens` (offset 0, 8B), `n_tokens` (offset 8, 4B), `start_pos` (offset 12, 4B), `seq_id` (offset 16, 4B), `request_logits` (offset 20, 1B), `reserved_padding[3]` (offset 21, 3B), `abort_flag` (offset 24, 8B).
* **`argus_multimodal_params_t` (16 Bytes, 8-Byte Aligned):** `mmproj_path` (offset 0, 8B), `cpu_threads` (offset 8, 4B), `use_gpu` (offset 12, 1B), `reserved_padding[3]` (offset 13, 3B).
* **`argus_logit_bias_t` (8 Bytes, 4-Byte Aligned):** `token` (offset 0, 4B), `bias` (offset 4, 4B).

---

## 3. M-RoPE, KV Cache Rollback & Sequence Slot Forking

### 3.1 M-RoPE & Automagic Prefix Rollback
* Vision-language architectures (such as Qwen2-VL and Qwen2.5-VL) use Multimodal Rotary Position Embeddings (M-RoPE / IM-RoPE), assigning 4 position coordinates per token (`model.isMRoPE() == true`, `model.nPosPerEmbd() == 4`).
* When `context.decodeBatch(...)` or `context.evalMultimodalChunks(...)` receives a `startPos <= context.getSeqPosMax(seqId)`, `libargus` automatically prunes invalidated KV cache cells in `[startPos, -1)` across both the primary context and any attached speculative draft context, and prunes coordinate-tagged sampler history entries with `kv_pos >= startPos` while preserving primed system tokens (`kv_pos == -1`).

### 3.2 Recurrent / Hybrid Topologies & Zero-Rollback Sequence Slot Forking
* Hybrid attention-recurrent architectures (e.g. `LLM_ARCH_QWEN35`, `model.isHybrid() == true`, `model.isRecurrent() == true`) maintain recurrent state tensors that cannot be partially rolled back ($X < Y$). Attempting a partial rollback returns `ArgusContext.STATUS_ROLLBACK_FAILED` (`-3` / `ARGUS_ERROR_ROLLBACK_FAILED`) cleanly without corrupting sampler history.
* To explore divergent reasoning branches or reuse a shared system prompt on hybrid models with zero additional KV memory (`kvUnified = true`), configure `.cloneSlots(N)` on `ArgusContextConfig.Builder` and use `int workerSlot = context.forkSlot(0)` / `context.freeSlot(workerSlot)`.

---

## 4. Zero-Allocation Steady-State Hot-Path & Heap Profiling

`libargus` is engineered for zero per-token Java heap allocations on steady-state `MemorySegment` / `DirectByteBuffer` decode and persistent-sampler paths after JIT warmup. Off-path setup, model loading, string conversions (`generate(...)`), and multimodal container parsing allocate standard JVM or native memory as needed.

You can measure per-call Java heap allocation rates on your target JDK using [`scripts/AllocationProbe.java`](../scripts/AllocationProbe.java):

```bash
java --enable-native-access=ALL-UNNAMED \
  -Dcc.projectargus.libargus.path="$PWD/build/lib/libargus.so" \
  -cp bindings/java/libargus-common/build/libs/libargus-common-2.3.0.jar:bindings/java/libargus-core/build/libs/libargus-core-2.3.0.jar \
  scripts/AllocationProbe.java tests/data
```

---

## 5. Native C++ Build & Hardware Acceleration Matrix

`libargus` uses CMake `FetchContent` to statically link pinned upstream `llama.cpp` (`libmtmd`) and `whisper.cpp` compute cores into a single shared binary (`libargus.so`, `argus.dll`, or `libargus.dylib`).

```bash
# 1. Portable CPU Baseline (x86-64-v3 AVX2/BMI2/FMA/F16C or ARMv8-A)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON
cmake --build build --config Release -j $(nproc)

# 2. NVIDIA CUDA Acceleration
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# 3. AMD ROCm / HIP Acceleration
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_HIP=ON
cmake --build build --config Release -j $(nproc)

# 4. Cross-Platform Vulkan Acceleration
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)

# 5. Apple Silicon Metal Acceleration (macOS)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_METAL=ON
cmake --build build --config Release -j $(sysctl -n hw.ncpu)

# Or trigger CMake directly via Gradle:
./gradlew compileCMake -Pcuda=true
./gradlew compileCMake -Phip=true
./gradlew compileCMake -Pvulkan=true
./gradlew compileCMake -Pmetal=true
```
