# LangChain4j Developer Cookbook (`:libargus-langchain4j`)

`:libargus-langchain4j` provides an in-process, zero-HTTP-overhead [LangChain4j `1.23.0`](https://docs.langchain4j.dev/) integration backed directly by `libargus`. It implements `ChatModel`, `StreamingChatModel`, `EmbeddingModel`, `AudioTranscriptionModel`, `TextToSpeechModel`, and `TokenCountEstimator` over the binding-neutral `cc.projectargus.libargus.*` Java surface, allowing you to pair it with either:

* **`:libargus-core` (JDK 22+ Project Panama FFM):** Maximum desktop/server throughput with zero JNI transition overhead.
* **`:libargus-jni` (JDK 17+ / Android API 26+):** Drop-in compatibility for JDK 17/21 LTS servers and Android devices.

> [!TIP]
> **Looking for a full runnable Spring Boot 3.5 + LangChain4j 1.23.0 application?**
> Check out **[libargus.cc-spring-boot-example](https://github.com/ProjectArgus-cc/libargus.cc-spring-boot-example)**—a standalone Local AI Workbench showcasing shared `ArgusModel` weights, `@Bean` wiring, incremental SSE token & `<think>` reasoning streaming, Multimodal Vision (`mmproj`), local Whisper ASR & Speech-LLM TTS, `StreamingHandle.cancel()` + `ArgusAbortFlag` cancellation, exact BPE/vision-patch budgeting, and automatic JDK 25 / 22+ Panama FFM vs. JDK 17+ JNI selection.

---

## 1. Gradle Dependency Setup

### Option A — Server / Desktop (JDK 22+ with Project Panama FFM)
```kotlin
dependencies {
    implementation("cc.projectargus:libargus-langchain4j:2.4.0")
    implementation("cc.projectargus:libargus-core:2.4.0")

    // Zero-config native binary extraction for your target OS/CPU/GPU:
    runtimeOnly("cc.projectargus:libargus-native-linux-cuda:2.4.0")
}
```

### Option B — JDK 17 / 21 LTS or Android (Standard JNI)
```kotlin
dependencies {
    implementation("cc.projectargus:libargus-langchain4j:2.4.0")
    implementation("cc.projectargus:libargus-jni:2.4.0")

    // Choose your platform runtime classifier (e.g., Linux CPU, macOS Metal, Windows Vulkan):
    runtimeOnly("cc.projectargus:libargus-native-linux-cpu:2.4.0")
}
```

---

## 2. Synchronous Chat, Reasoning (`<think>`) & Per-Request Overrides (`LibArgusChatModel`)

`LibArgusChatModel` implements LangChain4j's `ChatModel` interface, supporting multi-turn conversation formatting (`LibArgusChatFormatter.autoDetect()`, `chatMl()`, `llama3()`, `gemma()`, `mistral()`, or custom `LibArgusChatFormatter`), opt-in `<think>...</think>` reasoning trace separation (`.returnThinking(true)` $\to$ `aiMessage().thinking()`), `ChatModelListener` observability, per-request hyperparameter overrides (`ChatRequestParameters`), BPE-boundary-safe stop sequences, and exact token usage telemetry (`TokenUsage`, `FinishReason`).

```java
// snippet-id: sync-chat-overrides
import cc.projectargus.libargus.langchain4j.LibArgusChatFormatter;
import cc.projectargus.libargus.langchain4j.LibArgusChatModel;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.util.List;

public class SyncChatExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf");

        try (LibArgusChatModel model = LibArgusChatModel.builder()
                .modelPath(modelPath)
                .chatFormatter(LibArgusChatFormatter.chatMl())
                .returnThinking(true)
                .contextLength(8192)
                .gpuLayers(0)
                .temperature(0.7f)
                .maxOutputTokens(256)
                .seed(42)
                .build()) {

            ChatRequest request = ChatRequest.builder()
                    .messages(
                        SystemMessage.from("You are a principal distributed systems architect. Be concise."),
                        UserMessage.from("Why does in-process FFI beat loopback HTTP for local LLM inference?")
                    )
                    .parameters(ChatRequestParameters.builder()
                        .temperature(0.0)
                        .maxOutputTokens(16)
                        .stopSequences(List.of("\n\n"))
                        .build())
                    .build();

            ChatResponse response = model.chat(request);
            if (response.aiMessage().thinking() != null) {
                System.out.println("Thinking: " + response.aiMessage().thinking());
            }
            System.out.println("Assistant: " + response.aiMessage().text());
            System.out.printf("Tokens (in/out/total): %d/%d/%d | Stop: %s%n",
                    response.tokenUsage().inputTokenCount(),
                    response.tokenUsage().outputTokenCount(),
                    response.tokenUsage().totalTokenCount(),
                    response.finishReason());
        }
    }
}
```

---

## 3. Live Token Streaming & `StreamingHandle` Cancellation (`LibArgusStreamingChatModel`)

`LibArgusStreamingChatModel` implements `StreamingChatModel`, streaming decoded UTF-8 pieces to `StreamingChatResponseHandler` (`onPartialResponse` and `onPartialThinking`) while holding back partial stop-sequence and `<think>` prefixes. Both `context.streamingHandle().cancel()` and reactive `Flow.Subscription.cancel()` (enabled via `.asyncExecutor(Executor)`) wire directly into native `ArgusAbortFlag` cooperative cancellation.

```java
// snippet-id: streaming-chat-abort
import cc.projectargus.libargus.langchain4j.LibArgusStreamingChatModel;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class StreamingChatExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf");

        try (LibArgusStreamingChatModel streamModel = LibArgusStreamingChatModel.builder()
                .modelPath(modelPath)
                .returnThinking(true)
                .contextLength(8192)
                .gpuLayers(0)
                .maxOutputTokens(256)
                .seed(42)
                .build()) {

            CompletableFuture<ChatResponse> done = new CompletableFuture<>();
            ChatRequest request = ChatRequest.builder()
                    .messages(List.of(UserMessage.from("Write a haiku about CPU cache lines.")))
                    .build();

            streamModel.chat(request, new StreamingChatResponseHandler() {
                @Override
                public void onPartialThinking(PartialThinking partialThinking, PartialThinkingContext context) {
                    System.out.print(partialThinking.text());
                }

                @Override
                public void onPartialResponse(PartialResponse partialResponse, PartialResponseContext context) {
                    System.out.print(partialResponse.text());
                    System.out.flush();
                    // Call context.streamingHandle().cancel() at any time to abort native decoding
                }

                @Override
                public void onCompleteResponse(ChatResponse response) {
                    System.out.println("\n[Finished: " + response.finishReason() + "]");
                    done.complete(response);
                }

                @Override
                public void onError(Throwable error) {
                    done.completeExceptionally(error);
                }
            });

            done.join();
        }
    }
}
```

---

## 4. Bidirectional Document Embeddings & Matryoshka RAG (`LibArgusEmbeddingModel`)

`LibArgusEmbeddingModel` implements `EmbeddingModel` with L2-normalized vector extraction, asymmetric `EmbeddingInputType` (`QUERY` vs. `DOCUMENT`) prefixing, Matryoshka `EmbeddingRequest.dimensions()` slicing, `EmbeddingModelListener` observability, and cumulative `TokenUsage` tracking. Setting `.causalAttn(false)` preserves bidirectional non-causal attention masks required by BERT, `EmbeddingGemma 2`, and `jina-embeddings-v3`, and automatically sizes `uBatch` to `contextLength`.

```java
// snippet-id: rag-embeddings
import cc.projectargus.libargus.langchain4j.LibArgusEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.output.Response;
import java.nio.file.Path;
import java.util.List;

public class EmbeddingRagExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/jina-embeddings-v3-Q4_K_M.gguf");

        try (LibArgusEmbeddingModel embeddingModel = LibArgusEmbeddingModel.builder()
                .modelPath(modelPath)
                .contextLength(512)
                .gpuLayers(0)
                .causalAttn(false) // Preserves bidirectional non-causal attention masks
                .queryPrefix("search_query: ")
                .documentPrefix("search_document: ")
                .normalize(true)
                .build()) {

            List<TextSegment> docs = List.of(
                TextSegment.from("libargus unifies LLM, Whisper ASR, TTS, and Vision in one C++17 binary."),
                TextSegment.from("ArgusSequenceSlotPool partitions primary and worker clone KV cache slots.")
            );

            Response<List<Embedding>> result = embeddingModel.embedAll(docs);
            EmbeddingResponse queryResp = embeddingModel.embed(EmbeddingRequest.builder()
                    .input("How does libargus execute Whisper ASR and LLM inference?")
                    .inputType(EmbeddingInputType.QUERY)
                    .build());

            float[] v0 = result.content().get(0).vector();
            float[] v1 = result.content().get(1).vector();
            double dot = 0.0;
            for (int i = 0; i < v0.length; i++) {
                dot += (double) v0[i] * v1[i];
            }
            System.out.printf("Dimension: %d | Segments: %d | Query tokens: %d%n",
                    result.content().get(0).dimension(),
                    result.content().size(),
                    queryResp.tokenUsage().inputTokenCount());
            System.out.printf("Cosine similarity (doc0 vs doc1): %.4f%n", dot);
        }
    }
}
```

---

## 5. Zero-Trust Multimodal Vision, Audio & Video Chat (`ImageContent`, `AudioContent` & `VideoContent`)

When `.mmprojPath(...)` is configured on `LibArgusChatModel` or `LibArgusStreamingChatModel`, `UserMessage` instances containing `ImageContent`, `AudioContent`, and `VideoContent` are automatically projected via `ArgusMultimodalContext` in-process (with bounded `.videoFps(...)` and `.maxVideoFrames(...)` extraction for video streams).

> [!IMPORTANT]
> **Zero-Trust Media Sandbox Policy:**
> * **Inline Binary / Base64 (`ImageContent` / `AudioContent` / `VideoContent`):** Enabled by default and guarded by a pre-decode `.maxMediaBytes(...)` cap (default `64 MiB`) prior to Java heap allocation.
> * **Remote URLs (`http://`, `https://`):** Unconditionally rejected to prevent Server-Side Request Forgery (SSRF).
> * **Local `file:` URIs:** Rejected by default unless confined to a canonical sandbox directory via `.allowedMediaRoot(Path)` (which resolves `toRealPath()` to block `../` traversal and symlink escapes) or explicitly enabled via `.allowLocalFileMedia(true)`.

```java
// snippet-id: multimodal-vision-audio
import cc.projectargus.libargus.langchain4j.LibArgusChatModel;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;

public class MultimodalChatExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/qwen2-vl-7b-it.gguf");
        Path mmprojPath = Path.of(args.length > 1 ? args[1] : "models/qwen2-vl-7b-it.mmproj");
        Path mediaSandbox = Path.of(args.length > 2 ? args[2] : "media");
        Path imagePath = args.length > 3 ? Path.of(args[3]) : mediaSandbox.resolve("rack-thermal.png");
        Path audioPath = args.length > 4 ? Path.of(args[4]) : mediaSandbox.resolve("alarm-capture.wav");

        try (LibArgusChatModel visionModel = LibArgusChatModel.builder()
                .modelPath(modelPath)
                .mmprojPath(mmprojPath)
                .gpuLayers(0)
                .multimodalUseGpu(false)
                .allowedMediaRoot(mediaSandbox) // Confines file:// URIs to mediaSandbox
                .maxMediaBytes(32L * 1024 * 1024) // 32 MiB pre-decode cap
                .maxVideoFrames(16)
                .contextLength(8192)
                .maxOutputTokens(256)
                .build()) {

            UserMessage multimodalMsg = UserMessage.from(
                TextContent.from("What anomalies do you see in this server rack photo, and what does the audio alert say?"),
                ImageContent.from(imagePath.toUri()),
                AudioContent.from(audioPath.toUri())
            );

            ChatResponse response = visionModel.chat(multimodalMsg);
            System.out.println("Multimodal response: " + response.aiMessage().text());
        }
    }
}
```

---

## 6. Zero-Duplicate-VRAM Model Sharing (`ArgusModel` Shared Across Models & Token Estimator)

When passing an externally loaded `.model(ArgusModel)` to `LibArgusChatModel`, `LibArgusStreamingChatModel`, `LibArgusEmbeddingModel`, `LibArgusTextToSpeechModel`, or `LibArgusTokenCountEstimator`, closing the LangChain4j wrapper closes only its dedicated `ArgusContext` while leaving the shared `ArgusModel` weights resident in memory.

```java
// snippet-id: shared-model-lifecycle
import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.langchain4j.LibArgusChatModel;
import cc.projectargus.libargus.langchain4j.LibArgusTokenCountEstimator;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import java.nio.file.Path;
import java.util.List;

public class SharedModelExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf");
        ArgusBackend.init();

        // Load GGUF weights once; share across deterministic + creative ChatModels and TokenCountEstimator
        try (ArgusModel sharedWeights = ArgusModel.load(modelPath, 0, false);
             LibArgusChatModel fastDeterministicChat = LibArgusChatModel.builder()
                 .model(sharedWeights)
                 .contextLength(4096)
                 .temperature(0.0f)
                 .maxOutputTokens(128)
                 .build();
             LibArgusChatModel creativeBrainstormChat = LibArgusChatModel.builder()
                 .model(sharedWeights)
                 .contextLength(4096)
                 .temperature(0.8f)
                 .maxOutputTokens(128)
                 .build();
             LibArgusTokenCountEstimator estimator = LibArgusTokenCountEstimator.fromModel(sharedWeights)) {

            int promptBudget = estimator.estimateTokenCountInMessages(List.of(
                SystemMessage.from("You are a helpful assistant."),
                UserMessage.from("Check prompt budget before running inference.")
            ));
            System.out.println("Prompt token count: " + promptBudget);

            System.out.println("Fast reply: " + fastDeterministicChat.chat("Hello!"));
            System.out.println("Creative reply: " + creativeBrainstormChat.chat("Hello!"));
        }
    }
}
```

---

## 7. In-Process Speech Synthesis (TTS) & Whisper Transcription (ASR)

`LibArgusTextToSpeechModel` (`TextToSpeechModel`) synthesizes canonical `audio/wav` payloads via Speech-LLM + WavTokenizer vocoder DSP, while `LibArgusAudioTranscriptionModel` (`AudioTranscriptionModel`) decodes multi-rate/multi-channel RIFF/WAVE or raw PCM audio on Android and desktop/server JVMs without `javax.sound.sampled` and transcribes via `ArgusAudioContext` (Whisper).

```java
// snippet-id: speech-tts-and-whisper-asr
import cc.projectargus.libargus.langchain4j.LibArgusAudioTranscriptionModel;
import cc.projectargus.libargus.langchain4j.LibArgusTextToSpeechModel;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import dev.langchain4j.model.audio.TextToSpeechResponse;
import java.nio.file.Path;

public class SpeechPipelineExample {
    public static void main(String[] args) {
        Path ttsModelPath = Path.of(args.length > 0 ? args[0] : "models/outetts-0.2-500m.gguf");
        Path vocoderPath = Path.of(args.length > 1 ? args[1] : "models/wavtokenizer-large-75token.gguf");
        Path whisperPath = Path.of(args.length > 2 ? args[2] : "models/whisper-base.en.gguf");

        try (LibArgusTextToSpeechModel tts = LibArgusTextToSpeechModel.builder()
                .modelPath(ttsModelPath)
                .vocoderModelPath(vocoderPath)
                .gpuLayers(0)
                .contextLength(4096)
                .voice("athena", 42)
                .build();
             LibArgusAudioTranscriptionModel whisper = LibArgusAudioTranscriptionModel.builder()
                .modelPath(whisperPath)
                .gpuLayers(0)
                .trimOutput(true)
                .build()) {

            TextToSpeechResponse synthesized = tts.synthesize(
                TextToSpeechRequest.builder()
                    .text("Welcome to Project Argus.")
                    .voice("athena")
                    .build()
            );
            Audio wavAudio = synthesized.audio();
            String transcript = whisper.transcribeToText(wavAudio);
            System.out.printf("WAV bytes: %d | Transcript: %s%n",
                    wavAudio.binaryData().length,
                    transcript);
        }
    }
}
```

---

## 8. `1.23.0` Observability, Video Bounds & GPU-Pinned Async Reference

| Capability | Builder / Request API | Architectural Behavior |
| :--- | :--- | :--- |
| **`<think>` Reasoning Traces** | `.returnThinking(true)` (`false` by default) | Separates `<think>...</think>` (and prompt-suffixed `<think>`) via `LibArgusThinkingParser` into `AiMessage.thinking()` and `onPartialThinking(...)` with zero BPE tag-fragment leaks. |
| **Observability Listeners** | `.addListener(ChatModelListener)` / `.addListener(EmbeddingModelListener)` | Invokes LangChain4j `onRequest`, `onResponse` (with `modelName` & `TokenUsage`), and `onError` hooks across sync, streaming, and async execution. |
| **GPU-Pinned Async & Reactive `Flow.Publisher`** | `.asyncExecutor(Executor)` (`null` by default) | Enables `chatAsync(...)`, `embedAsync(...)`, and demand-aware `StreamingChatModel.chat(ChatRequest)` (`Flow.Publisher<ChatModelStreamingEvent>`) with `Subscription.cancel()` wired into `ArgusAbortFlag`. Intentionally throws/emits `AsyncNotSupportedException` when unset rather than defaulting to `ForkJoinPool.commonPool()`, preserving CUDA thread-local (`cublasHandle_t`) carrier affinity. |
| **Asymmetric & Matryoshka Embeddings** | `EmbeddingRequest.builder().inputType(QUERY \| DOCUMENT).dimensions(d)` | Selects `.queryPrefix(...)` vs. `.documentPrefix(...)` per request (or per `TextSegment` via `argus.embedding_type`) and slices + L2-renormalizes Matryoshka vectors in `[1, nEmbd]`. |
| **Bounded `VideoContent` Projection** | `.videoFps(1.0f)`, `.videoTimestampIntervalMs(1000L)`, `.maxVideoFrames(16)` | Extracts bounded video frames and timestamp markers via `ArgusVideo` into 1:1 `<__media__>` projector slots while escaping untrusted `<__media__>` literals inside user text. |
| **Sub-Second Whisper Utterances** | `.padShortAudioToOneSecond(true)` (`true` by default) | Zero-pads `< 1.0 s` (`< 16,000` sample) commands (`"Yes"`, `"Stop"`) with trailing silence so `whisper.cpp` never drops sub-second utterances. |
