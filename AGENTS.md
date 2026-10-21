# AGENTS.md — Core Operational Directives & Architectural Contract for `libargus`

This document defines the cognitive stance, non-negotiable architectural invariants, Layer-0 $\leftrightarrow$ Layer-1 downstream contracts, module boundaries, concurrency locks, ABI wire layouts, and verification gates for `libargus`. Every AI coding agent and human maintainer modifying this repository **must** preserve these invariants.

---

## 0. AI Persona & Cognitive Stance

When acting within this repository, you adopt the mindset of a **Principal Low-Level Systems Architect, C++17 Native Runtime Engineer, and Core JVM FFI / Mechanical Sympathy Specialist** with deep expertise in zero-allocation off-heap execution, lock-free/fine-grained concurrency, and Project Panama / JNI runtimes.

### Core Cognitive Triggers
* **Bare-Metal & Zero-Allocation Sympathy:** Evaluate every change through the lens of heap efficiency, CPU cache lines, safepoint overhead, and off-heap memory lifecycles. Steady-state token decoding, persistent sampler evaluation, and `ArgusVideoItem` scratch recycling must operate with **zero per-token Java heap allocations** after JIT warmup (verified via [`scripts/AllocationProbe.java`](scripts/AllocationProbe.java)).
* **1:1 ABI & Struct Packing Precision:** Public C structures use explicit `reserved_padding` bytes with zero compiler-injected alignment holes (e.g., 56-byte `argus_sampler_params_t` at 4-byte alignment, 56-byte `argus_perf_timings_t` at 8-byte alignment, 8-byte `argus_logit_bias_t` at 4-byte alignment). Every layout is verified at compile time in [`src/argus_abi.h`](src/argus_abi.h) (`static_assert`) and introspectable at runtime via `ArgusBackend.getBuildInfo()`.
* **Adversarial FFI Boundary Stance:** Assume an unforgiving, concurrent multi-tenant caller environment across the Java-to-C boundary. Every `MemorySegment`, `DirectByteBuffer`, primitive array slice, and raw pointer must pass preflight bounds, liveness, alignment, and 64-bit integer-overflow checks (`ArgusCommonValidation`, `ArgusValidation`, `ArgusJniValidation`) before crossing into native execution. Every fallible C export must be wrapped in a `try/catch` exception barrier capturing diagnostics in thread-local storage (`argus_last_error_code`, `argus_last_error_message_copy`).
* **Strict `:libargus-common` DRY & 100% Dual-Binding Symmetry:** Enforce single-responsibility, reusable primitives in `:libargus-common` (`--release 17`) on the first pass (`ArgusSequenceSlotPool`, `ArgusTextGenerator`, `ArgusCommonValidation`, `ArgusUtf8StreamDecoder`, `ArgusNativeLoader`, `AbstractNativeLibraryProvider`). Never duplicate pure-Java state machines or validation logic between `:libargus-core` and `:libargus-jni`. Every public class and method in `:libargus-core` (JDK 22+ Panama FFM) **must** have an exact counterpart in `:libargus-jni` (JDK 17+ / Android) and a fanged differential test in [`ArgusDifferentialEquivalenceTest`](bindings/java/libargus-jni/src/test/java/cc/projectargus/libargus/ArgusDifferentialEquivalenceTest.java).
* **Fanged Testing & Fault-Injection Mandate:** Zero tautological tests or vanity coverage metrics. Every concurrency lock, handle lease, bounds check, and rollback path must possess "fangs"—meaning deterministic state-mutation assertions that fail when the underlying protection is removed by [`scripts/release/mutation_check.py`](scripts/release/mutation_check.py).

---

## 1. System Vision & Layer-0 $\leftrightarrow$ Layer-1 Downstream Contract

`libargus` is **Layer 0 (The Core Execution Bedrock)** for the broader Project Argus ecosystem. It consolidates LLM text generation, Whisper speech-to-text (ASR), Speech-LLM text-to-speech (TTS), and `libmtmd` Multimodal (Vision, Audio, and Video) pipelines into a single process-global C++17 native runtime consumed by Project Argus's Layer-1 stateful cognitive core (`L-TABB`).

### Downstream Layer-1 Integration Invariants
Modifications to `libargus` must never break the execution contracts relied upon by downstream Layer-1 orchestration:
1. **VRAM Byte Budgeting & Attention Width Geometry (`n_embd_inp` vs. `n_embd_out`):**
   * Downstream resource schedulers budget VRAM strictly in bytes via `argus_model_kv_bytes_per_token`, `argus_model_estimate_vram_bytes`, `argus_model_size`, `quantTypeSize`, and `quantBlockSize`.
   * Models such as `EmbeddingGemma 2` have an internal backbone attention width (`n_embd_inp = 2048`) that differs from their output projection dimension (`n_embd_out = 768`, returned by `nEmbd()`). KV cache sizing **must** resolve against internal attention geometry, and `argus_perf_timings_t` must preserve the 56-byte layout including `n_reallocs_same` and `n_reallocs_diff` graph reallocation counters.
2. **Live CPU Thread Scaling:**
   * `argus_set_n_threads`, `argus_get_n_threads`, `argus_get_n_threads_batch`, and `argus_audio_set_n_threads` allow downstream schedulers to dynamically retune single-token generation and batch prefill CPU thread counts on active contexts without tearing down contexts or invalidating warm KV caches.
3. **M-RoPE / Hybrid-Recurrent Defense & Clone-Pool Partitioning:**
   * Hybrid attention-recurrent architectures (e.g., Qwen 3.5/3.8 `LLM_ARCH_QWEN35`) cannot perform arbitrary partial KV rollbacks without triggering upstream $X < Y$ assertion aborts.
   * `libargus` exposes topology introspection (`isHybrid()`, `isRecurrent()`, `isDiffusion()`, `canShift()`, `isMRoPE()`, `nPosPerEmbd()`) and intercepts unsupported partial rollbacks across text and multimodal evaluation, returning `ARGUS_DECODE_ROLLBACK_FAILED (-3)` / `ARGUS_ERROR_ROLLBACK_FAILED` without corrupting sampler history.
   * `ArgusContextConfig` and `ArgusSequenceSlotPool` partition sequence slots into primary slots `[0, seqMax)` and dedicated worker clone slots `[seqMax, effectiveSeqMax)` (`effectiveSeqMax = Math.max(1, seqMax) + cloneSlots`), enabling zero-rollback `forkSlot(srcSeqId)` branching with `kv_unified = true`.
4. **Bidirectional Non-Causal Embeddings (`causalAttn = false`):**
   * `ArgusContextConfig.Builder.causalAttn(false)` and atomic `evalEmbeddings` / `extractEmbeddings` must preserve bidirectional non-causal attention masks required by BERT, `EmbeddingGemma 2`, and `jina-embeddings-v3` models.
5. **Cross-Thread Dispatch & Shared Arena Safety:**
   * Downstream runtimes may pin native GPU inference calls to dedicated platform threads (preventing virtual-thread carrier hopping across thread-local `cublasHandle_t` pools) while multi-threaded worker pools concurrently access shared contexts. `ArgusContext` in `:libargus-core` manages an internal `Arena.ofShared()` guarded by its `ReentrantReadWriteLock` lease so cross-thread downcalls never throw `WrongThreadException`.

---

## 2. Codebase Topology & Module Boundaries

```text
libargus/
├── CMakeLists.txt                   # Layer-0 C++17 build, portable ISA flags & SHA-pinned FetchContent
├── version.txt                      # Single source of truth for semantic version (2.1.1)
├── include/
│   └── libargus.h                   # Master C ABI stable layout & export definitions
├── src/
│   ├── argus_abi.h                  # Compile-time static_assert layout, size & offset verification
│   ├── argus_internal.h             # Private structs (argus_model, argus_context, sampler state)
│   ├── argus_execution_lock.h       # Per-resource mutex & test concurrency observer hooks
│   ├── argus_dsp.h                  # Bounds-checked overlap-add vocoder DSP math
│   ├── argus_common.cc              # Process-global backend lifecycle, logging, TLS diagnostics
│   ├── argus_text.cc                # Llama model/FD loading, decode, sampler state machine, TTS
│   ├── argus_audio.cc               # Whisper ASR contexts & bounded transcription
│   ├── argus_multimodal.cc          # libmtmd vision/audio projectors & FFmpeg video pipes
│   └── jni/
│       └── argus_jni.cc             # Unified JNI bridge compiled directly into libargus
├── bindings/java/
│   ├── libargus-common/             # Shared JDK 17+ (--release 17) records, validation, SPI & drivers
│   │   └── src/main/java/cc/projectargus/libargus/
│   │       ├── ArgusSamplerConfig.java, ArgusPerfTimings.java, ArgusBuildInfo.java
│   │       ├── ArgusLogLevel.java, ArgusLogCallback.java, ArgusNativeException.java
│   │       ├── internal/
│   │       │   ├── AbstractNativeResource.java     # ReentrantReadWriteLock handle lease base
│   │       │   ├── ArgusCommonValidation.java      # Shared KV constants, slice & slot bounds checks
│   │       │   ├── ArgusNativeLoader.java          # SPI + path/env native library loader
│   │       │   ├── ArgusSequenceSlotPool.java      # Thread-safe BitSet sequence slot lease pool
│   │       │   ├── ArgusTextGenerator.java         # Unified generate, generateUntil & extractEmbeddings
│   │       │   └── ArgusUtf8StreamDecoder.java     # Stateful multi-byte UTF-8 streaming decoder
│   │       └── spi/
│   │           ├── NativeLibraryProvider.java      # Platform SPI interface
│   │           └── AbstractNativeLibraryProvider.java # DRY classpath extraction base class
│   ├── libargus-core/               # JDK 22+ (--release 22) Project Panama FFM binding module
│   │   └── src/main/java/cc/projectargus/libargus/
│   │       ├── ArgusBackend.java, ArgusModel.java, ArgusContext.java, ArgusContextConfig.java
│   │       ├── ArgusAudioContext.java, ArgusMultimodalContext.java
│   │       ├── ArgusBitmap.java, ArgusVideo.java, ArgusVideoItem.java
│   │       ├── ArgusInputChunks.java, ArgusAbortFlag.java
│   │       └── internal/ (ArgusLayouts, ArgusBindings, ArgusNativeResource, ArgusValidation)
│   ├── libargus-jni/                # JDK 17+ / Android (--release 17) drop-in JNI binding module
│   │   └── src/main/java/cc/projectargus/libargus/
│   │       ├── ArgusBackend.java, ArgusModel.java, ArgusContext.java, ArgusContextConfig.java
│   │       ├── ArgusAudioContext.java, ArgusMultimodalContext.java
│   │       ├── ArgusBitmap.java, ArgusVideo.java, ArgusVideoItem.java
│   │       ├── ArgusInputChunks.java, ArgusAbortFlag.java
│   │       └── internal/ (ArgusJni, ArgusJniResource, ArgusJniValidation)
│   └── libargus-native-*/           # 8 platform SPI runtime classifier modules (--release 17)
├── tests/                           # C++17 CTest integration, media & concurrency suites
└── scripts/
    ├── AllocationProbe.java         # Zero-allocation steady-state heap profiler
    ├── ci/                          # Workflow linting & CI helper scripts
    └── release/                     # Deterministic release candidate, pin & mutation verifiers
```

### Java Module Compilation Rules
* **`:libargus-common` (`--release 17`):** Contains zero Panama (`java.lang.foreign.*`) or JNI `native` method declarations. Houses all shared records, builders, validation helpers, `AbstractNativeResource`, `ArgusSequenceSlotPool`, `ArgusTextGenerator`, `ArgusUtf8StreamDecoder`, `ArgusNativeLoader`, and SPI contracts (`NativeLibraryProvider`, `AbstractNativeLibraryProvider`).
* **`:libargus-core` (`--release 22`):** Uses Project Panama FFM (`MemorySegment`, `Arena`, `Linker`) while supporting primitive array and slice convenience overloads for API parity with `:libargus-jni`.
* **`:libargus-jni` (`--release 17`):** Uses standard JNI (`long` native handles, `DirectByteBuffer`, and primitive array/slice overloads) for JDK 17+ and Android API 26+ runtimes.
* **100% Subsystem Parity Rule:** Every public class (`ArgusBackend`, `ArgusModel`, `ArgusContext`, `ArgusContextConfig`, `ArgusAudioContext`, `ArgusMultimodalContext`, `ArgusBitmap`, `ArgusVideo`, `ArgusVideoItem`, `ArgusInputChunks`, `ArgusAbortFlag`) lives in package `cc.projectargus.libargus` in both `:libargus-core` and `:libargus-jni`. Any new high-level convenience API added to one binding module **must** be added to the other and tested in [`ArgusDifferentialEquivalenceTest`](bindings/java/libargus-jni/src/test/java/cc/projectargus/libargus/ArgusDifferentialEquivalenceTest.java).

---

## 3. Non-Negotiable Memory, Concurrency & FFI Invariants

### 3.1 Single-Binary Native Architecture & Process-Global Backend Singularity
* Both the C ABI (`argus_*`) and JNI entrypoints (`Java_cc_projectargus_libargus_jni_ArgusJni_*`) compile into a **single** shared library (`libargus.so` / `argus.dll` / `libargus.dylib`). Never split JNI into a separate shared library; doing so causes `dlopen` symbol resolution races on Android Bionic linker namespaces and desktop JVMs.
* `argus_backend_init()` orchestrates a singular, process-global hardware initialization pathway (`ggml_backend_load_all()`) across text, audio, speech, and multimodal subsystems to eliminate VRAM fragmentation and multi-context driver race conditions.

### 3.2 Native Handle Leasing (`AbstractNativeResource`) & Shared Arena Concurrency
* All Java wrappers around unmanaged pointers extend `AbstractNativeResource` (`ArgusNativeResource` for `MemorySegment` in `:libargus-core`, `ArgusJniResource` for `long` in `:libargus-jni`), backed by a `ReentrantReadWriteLock`.
* Every native downcall must execute under a shared read lease (`try (var lease = res.lease())` or `withHandle(...)`), while `close()` acquires the exclusive write lock to zero out and release the native handle once.
* Composite operations (`ArgusContext.init`, `ArgusContext.decodeBatch` with `ArgusAbortFlag`, `ArgusContext.evalMultimodalChunks`, `ArgusContext.synthesizeSpeech`, `ArgusMultimodalContext.tokenize`, `ArgusBitmap.loadFile`, `ArgusVideo.loadFile`) **must** hold read leases on all participating wrapper arguments for the duration of the native call.
* In `:libargus-core`, `ArgusContext` owns an internal `Arena.ofShared()` protected by the resource lifecycle lock so multi-threaded worker pools can concurrently dispatch operations without tripping `WrongThreadException`.

### 3.3 Independent Context Ownership & Deferred Backend Teardown
* `argus_context_init` and `argus_multimodal_init` increment the native reference count of their `argus_model_t` (`argus_model_retain`). Closing the Java `ArgusModel` wrapper early must not invalidate active contexts.
* `ArgusContext` queries its retained native model via `argus_context_get_model(ctx)` under the context's read lease rather than acquiring a lease on `modelRef`.
* Active native models and contexts increment `g_backend_active_resources`; `argus_backend_free()` defers `ggml` teardown if live resources still exist.

### 3.4 Selective Concurrency Locking & Shared Projector Lock Hierarchy
* Stateful context operations (`argus_decode_batch`, `argus_sample_token_ext`, KV cache mutations, thread tuning) serialize per context via `argus_execution_lock` (`ctx->mtx`), whereas read-only model/vocabulary/metadata queries (`argus_tokenize`, `argus_token_to_piece`, `argus_vocab_*`, `argus_model_meta_*`) execute **lock-free** across concurrent threads.
* During multimodal chunk evaluation (`argus_eval_multimodal_chunks`), the native lock hierarchy acquires the **multimodal projector mutex (`mctx->mtx`) first, then the text context mutex (`ctx->mtx`)** through encoding and embedding consumption. **Never invert this lock order.**

### 3.5 Project Panama `CRITICAL_ALLOWLIST`
* In `ArgusBindings.java` (`:libargus-core`), `Linker.Option.critical(false)` is strictly restricted via `CRITICAL_ALLOWLIST` to **exactly 4** pure, O(1), non-blocking leaf functions:
  1. `argus_build_features`
  2. `argus_abort_flag_is_requested`
  3. `argus_last_error_code`
  4. `argus_clear_error`
* Never attach `Linker.Option.critical(...)` to any function that locks a mutex, allocates memory, performs I/O, or invokes a callback/upcall.

### 3.6 GC-Friendly JNI Bridge & Standard UTF-8 Marshaling
* Never use `GetPrimitiveArrayCritical` during inference or I/O in `src/jni/argus_jni.cc` (prevents JVM GC pinning stalls). Use `Get*ArrayRegion` / `Set*ArrayRegion` or `GetDirectBufferAddress` for `DirectByteBuffer` overloads.
* Never use JNI Modified UTF-8 (`GetStringUTFChars` / `NewStringUTF`) for arbitrary text or token pieces, as it corrupts 4-byte supplementary Unicode (emojis/CJK) into CESU-8 surrogates. Always marshal strings as Standard UTF-8 `byte[]` slices (`StandardCharsets.UTF_8`) and decode streaming tokens via `ArgusUtf8StreamDecoder` (`accept` / `acceptUntil`).

### 3.7 File Descriptor Model Loading (`argus_model_load_fd`)
* `argus_model_load_fd(int32_t fd, const argus_model_params_t * params)` duplicates the caller's descriptor (`dup(fd)` on POSIX, `dup_win32_fd_into_static_crt(fd)` with `scoped_win32_crt_param_guard` on Windows), wraps the duplicate in a `FILE*`, unconditionally rewinds to offset `0` (`fseek(fp, 0, SEEK_SET)`), and closes the duplicated `FILE*` via RAII on all exit paths.
* Because the native layer duplicates `fd`, **the caller retains ownership of the original `fd`**. On Android, callers must pass `pfd.getFd()` inside a `try (ParcelFileDescriptor pfd = ...)` block—never `pfd.detachFd()` unless they manually close the detached descriptor.

### 3.8 Speculative & Multi-Token Prediction (MTP) Dual-Context Synchronization
* When a speculative draft model (`draftModel`) or Multi-Token Prediction (`draft-mtp`) is attached to an `ArgusContext`, primary and speculative draft KV caches (`ctx_draft`) must remain synchronized across `clearCacheSlot`, `shiftCacheSlot`, prefix rollbacks, and `setNThreads`.
* Single-sequence generation (`seq_max = 1`) allocates 100% of context memory to slot 0, while `kv_unified = true` enables dynamic cross-sequence KV cell sharing across speculative drafting, MTP tracks, and `forkSlot` worker clones.

### 3.9 Video Iteration & Scratch Carrier (`ArgusVideoItem`)
* `ArgusVideo` streams video through FFmpeg subprocesses into raw RGB frames and timestamp chunks (e.g., `"[00m05s]"`). `ArgusVideoItem` reuses its native scratch carrier across `video.readNext(item)` iterations; callers must not retain `item.bitmap()` after advancing or closing `ArgusVideoItem`.

### 3.10 Diagnostic Logging & Zero-Overhead Redirection
* `libargus` silences upstream C++ `stderr` noise by default (`ARGUS_LOG_WARN`) across `llama.cpp`, `ggml`, `whisper.cpp`, and `libmtmd`.
* Hot-path logging uses lock-free relaxed atomic severity checks, thread-local `ARGUS_LOG_CONT` continuation tracking, environment variable overrides (`LIBARGUS_LOG_LEVEL` or `ARGUS_LOG_LEVEL`), and exception-contained foreign callbacks (`ArgusBackend.setLogCallback`) across both Panama FFM upcalls and JNI global-ref handlers.

---

## 4. Sampler State Machine & Native Subsystem Invariants

| Subsystem / Contract | Invariant / Behavior | Native C ABI / Java Surface |
| :--- | :--- | :--- |
| **Continuation Commit** | Pending sample state machine. Normal continuation commits tokens directly to history without chain reset, replay, or RNG interruption. | `argus_decode_batch()`, `argus_sampler_has_pending()` |
| **Native Handle Leasing** | Read-write lock lease architecture prevents use-after-free and races between concurrent downcalls and `close()`. Decouples Java model handles from native refcounts. | `AbstractNativeResource`, `lease()` |
| **Panama Critical Downcalls** | Pure non-blocking C ABI leaf calls are bound with `Linker.Option.critical(false)`, eliminating JVM safepoint overhead on hot leaves. | `CRITICAL_ALLOWLIST` in `ArgusBindings` |
| **Rollback History Closure** | When `is_kv_rollback == true`, successfully decoded replacement tokens are committed to history and accepted into sampler chains. | `argus_rollback_to_kv_seq()` |
| **Logits Ownership** | Single-consumption model. Returns `-2` if target sequence was not evaluated last, if logits were already sampled, or if state was mutated. | `argus_sample_token_ext()` returns `-2` |
| **Coordinate Decoupling** | History entries are tagged with `kv_pos`. Primed prompt tokens (`kv_pos = -1`) survive KV rollbacks; only orphaned branch tokens (`kv_pos >= start_pos`) are pruned across text and multimodal rollbacks. | `argus_sampler_prime()`, `argus_decode_batch()`, `argus_eval_multimodal_chunks()` |
| **KV Position Shifting** | Relative position shifts (`delta`) update primary and speculative draft KV caches while shifting/pruning coordinate-linked `kv_pos` history in lockstep. | `argus_kv_cache_seq_add()`, `shiftCacheSlot()` |
| **Cancellation Handle Safety** | Reference-counted cancellation handles are retained natively via RAII throughout decode loops, preventing unmanaged use-after-free hazards. | `argus_abort_flag_t`, `ArgusAbortFlag` |
| **Entropy & RNG Seeding** | 32-bit seed (`0xFFFFFFFF` / `-1` for random entropy). Pure greedy argmax when `temperature <= 0.0f`. Distribution sampling when `temperature > 0.0f`. | `argus_sampler_params_t.seed` |
| **Monotonic RNG Continuity** | Dynamic hyperparameter tuning, partial KV rollbacks, and truncations clone the active distribution sampler RNG state without restarting to draw #0. | `llama_sampler_clone()` on reconfigure/rollback |
| **Multimodal Synchronization** | Per-context mutex serialization across chunk evaluation with synchronized KV cache and sampler history rollback pruning (`[n_past, -1)`). | `argus_eval_multimodal_chunks()` holds `ctx->mtx` |
| **C ABI Exception Barrier** | All native exports are wrapped in `try/catch` barriers; native errors are captured in thread-local storage and mapped to `ArgusNativeException`. | `argus_last_error_code()`, `ArgusNativeException` |
| **Bounded Diagnostics & Copy-Out** | Diagnostic retrieval, speech synthesis, and ASR transcription write into caller-bounded buffers with exact-fit capacity checks and incomplete-tail UTF-8 truncation. | `argus_last_error_message_copy()`, `argus_synthesize_speech_n()`, `argus_transcribe_audio()` |
| **Spatial Bounds Validation** | Preflight validation checks non-null addresses, buffer capacities, array slice bounds, and overflow-safe sizing prior to native downcalls. | `ArgusCommonValidation`, `ArgusValidation`, `ArgusJniValidation` |
| **Penalty History Replay** | Rebuilding filter chains on reconfiguration or rollback replays surviving sequence tokens via `llama_sampler_accept()`. | Persistent slot history buffer |
| **Stale Logits Invalidation** | Any KV cache mutation (`clearCacheSlot`, `shiftCacheSlot`), decode failure, multimodal projection error, TTS Sequence-0 cleanup, or slot reset immediately invalidates pending logits. | `invalidate_seq_logits()` |
| **History Introspection** | Zero-allocation real-time queries for retained token count (primed + committed) and uncommitted sample status. | `argus_sampler_get_history_count()`, `argus_sampler_has_pending()` |
| **Shared Projector Locking** | Native lock hierarchy acquires projector mutex **then** text context mutex through encoding and embedding consumption; never invert this lock order. | `argus_eval_multimodal_chunks()` |
| **Direct FFM Segment Safety** | Preflight validation rejects closed, heap-backed, wrong-thread-confined, short, or misaligned memory segments before unmanaged downcalls. | `ArgusValidation` |
| **Zero-Rollback Slot Forking** | KV cache replication combined with persistent sampler history replay and RNG distribution continuity across sequence tracks without native aborts. | `argus_kv_cache_seq_cp()`, `forkSlot()` |
| **Atomic Rollback Error Signaling** | Rejects unsupported partial cache rollbacks on recurrent architectures across text and multimodal evaluation, returning `-3` / `STATUS_ROLLBACK_FAILED` without desynchronizing sampler history. | `argus_decode_batch()`, `argus_eval_multimodal_chunks()`, `ARGUS_ERROR_ROLLBACK_FAILED` |
| **Structured ABI Introspection** | Zero-allocation query of version, source revisions, compiler, CPU baseline, feature masks, and wire layout offsets without backend initialization. | `ArgusBackend.getBuildInfo()` |

---

## 5. Native Memory Wire Layout & Project Panama Alignment

Public C structures in [`include/libargus.h`](include/libargus.h) and [`ArgusLayouts.java`](bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/internal/ArgusLayouts.java) use explicit `reserved_padding` bytes so compilers never inject implicit alignment gaps. Compile-time `static_assert` checks in [`src/argus_abi.h`](src/argus_abi.h) enforce exact byte offsets.

### 5.1 `argus_sampler_params_t` (56 Bytes, 4-Byte Aligned)
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

### 5.2 `argus_perf_timings_t` (56 Bytes, 8-Byte Aligned)
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

### 5.3 `argus_logit_bias_t` (8 Bytes, 4-Byte Aligned)
```text
Offset  Size  Type       Field Name            Description
---------------------------------------------------------------------------------------------
0       4     int32_t    token                 Target vocabulary token ID
4       4     float      bias                  Additive logit bias (-Float.MAX_VALUE bans token)
---------------------------------------------------------------------------------------------
Total Struct Byte Size: 8 bytes (0 implicit padding holes)
```

---

## 6. Build Matrix, Portable ISA Baselines & Verification Gates

### 6.1 Hardware Compilation Matrix & Portable Baselines
Distribution builds target **x86-64-v3** (AVX2, BMI2, FMA, F16C, SSE4.2) on Linux (Ubuntu 22.04 / glibc 2.35 baseline) and Windows (MSVC / Server 2022), and generic **ARMv8-A** on Apple Silicon (macOS 14+). `ARGUS_PORTABLE=ON` disables `-march=native` host tuning in both the wrapper and vendored GGML. External backend toolchains target CUDA 12.4.1, ROCm 6.1, Vulkan SDK, or system Metal.

```bash
# 1. Portable CPU Release Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON > /tmp/cmake_cpu.log 2>&1
cmake --build build --config Release -j $(nproc) >> /tmp/cmake_cpu.log 2>&1

# 2. Hardware Accelerator Targets (use a fresh build directory when switching toolchains)
cmake -B build-cuda   -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_CUDA=ON > /tmp/cmake_cuda.log 2>&1
cmake -B build-rocm   -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_HIP=ON > /tmp/cmake_rocm.log 2>&1
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DARGUS_PORTABLE=ON -DGGML_VULKAN=ON > /tmp/cmake_vk.log 2>&1

# Or invoke CMake via Gradle with accelerator flags:
./gradlew compileCMake -Phip=true > /tmp/gradle_hip.log 2>&1
./gradlew compileCMake -Pvulkan=true > /tmp/gradle_vk.log 2>&1
```

### 6.2 Build Native Test Library (`libargus_test.so`) & Run C++ CTest Suite
Always redirect verbose build/test logs to a file and run isolated failing tests before full suite runs. `ARGUS_MEDIA_REQUIRED=ON` requires `ffmpeg` and `ffprobe` on `PATH`.
```bash
cmake -B build -DARGUS_PORTABLE=ON -DARGUS_TESTING=ON -DARGUS_MEDIA_REQUIRED=ON > /tmp/cmake_cfg.log 2>&1
cmake --build build -j4 > /tmp/cmake_build.log 2>&1
ctest --test-dir build --output-on-failure > /tmp/ctest.log 2>&1
```

### 6.3 Run Java Unit & Integration Suites (`:libargus-common`, `:libargus-core`, `:libargus-jni`)
```bash
# Fast unit test pass for shared common internals (no native binary required):
./gradlew :libargus-common:test -PskipCMake=true > /tmp/test_common.log 2>&1

# Full Panama FFM + JNI + Differential Equivalence integration suite against libargus_test.so:
./gradlew :libargus-common:test :libargus-core:test :libargus-jni:test \
  -PskipCMake=true -PnativeTesting=true \
  -PnativePath="$PWD/build/lib/libargus_test.so" -PtestJdk=22 > /tmp/test_java.log 2>&1
```

### 6.4 Zero-Allocation Steady-State Heap Verification
Hot-path token decoding, persistent sampler evaluation, and `ArgusVideoItem` recycling are verified for zero steady-state Java heap allocations after JIT warmup via [`scripts/AllocationProbe.java`](scripts/AllocationProbe.java).

### 6.5 Run Release Tooling, Workflow Pin & Mutation Checks
```bash
python3 -m unittest discover -s scripts/release/tests -v > /tmp/test_release.log 2>&1
python3 scripts/release/pins.py > /tmp/pins.log 2>&1
python3 scripts/release/mutation_check.py native /tmp/argus-native-faults > /tmp/mut_native.log 2>&1
python3 scripts/release/mutation_check.py java /tmp/argus-java-faults > /tmp/mut_java.log 2>&1
```

---

## 7. Mutation Verification Anchors (`scripts/release/mutation_check.py`)

The fault-injection runner [`scripts/release/mutation_check.py`](scripts/release/mutation_check.py) validates that concurrency and lifecycle tests fail when specific locks or leases are removed in disposable source copies. **Do not alter or reformat these exact anchor strings:**

1. `src/argus_execution_lock.h`: `    if (!lock.try_lock()) {`
2. `src/argus_multimodal.cc`: `video->test_next++`, `--video->test_remaining;`, `    argus_multimodal_retain(mctx);`, and `        argus_test_allocation_checkpoint();`
3. `bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/ArgusContext.java`:
   ```java
   return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, lease.handle());
   ```

---

## 8. AI Assistant Constraints & Handoff Protocols

1. **Never Guess:** Inspect [`include/libargus.h`](include/libargus.h), [`src/argus_abi.h`](src/argus_abi.h), [`src/argus_internal.h`](src/argus_internal.h), and [`ArgusLayouts.java`](bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/internal/ArgusLayouts.java) directly whenever struct layouts, return codes, or handle lifecycles are in question.
2. **Maintain Single-Source Version Synchronization (`version.txt`):** To step the project version, update the single semantic version string inside [`/version.txt`](version.txt) at the repository root. CMake and Gradle read [`version.txt`](version.txt) automatically; never hardcode drifting version literals in build files or source code.
3. **Enforce Vendor Pin & Supply-Chain Discipline:** Never introduce floating git branches or unpinned tags in `CMakeLists.txt` (`FetchContent`) or `.github/workflows/*.yml`. All upstream commits, SHA-256 archive digests, and GitHub Action SHAs are verified by [`scripts/release/pins.py`](scripts/release/pins.py).
4. **Isolate Test Observer Hooks (`ARGUS_TESTING`):** Concurrency and fault-injection observer symbols (`argus_test_*`) must only compile when `-DARGUS_TESTING=ON` (`libargus_test.so`) and must never leak into production `libargus` release binaries.
5. **Eliminate Duplication Immediately (`:libargus-common` First):** Keep the codebase strictly **DRY** on the first pass—consolidate shared records, slot pools, text/UTF-8 streaming drivers, SPI loaders, and bounds validation into `:libargus-common` (`--release 17`) rather than duplicating logic across `:libargus-core`, `:libargus-jni`, or `:libargus-native-*`.
6. **Preserve Layer-0 $\leftrightarrow$ Layer-1 Compatibility:** Before altering any public C ABI or Java signature, verify that downstream Project Argus contracts (VRAM budgeting against `n_embd_inp`, live thread scaling, hybrid-recurrent `-3` rollback signaling, clone-pool slot partitioning, and non-causal embeddings) remain intact.
