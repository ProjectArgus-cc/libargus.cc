# LangChain4j Developer Cookbook (`:libargus-langchain4j`)

`:libargus-langchain4j` provides an in-process, zero-HTTP-overhead [LangChain4j](https://docs.langchain4j.dev/) integration backed directly by `libargus`. It implements `ChatModel`, `StreamingChatModel`, `EmbeddingModel`, and `TokenCountEstimator` over the binding-neutral `cc.projectargus.libargus.*` Java surface, allowing you to pair it with either:

* **`:libargus-core` (JDK 22+ Project Panama FFM):** Maximum desktop/server throughput with zero JNI transition overhead.
* **`:libargus-jni` (JDK 17+ / Android API 26+):** Drop-in compatibility for JDK 17/21 LTS servers and Android devices.

---

## 1. Gradle Dependency Setup

### Option A — Server / Desktop (JDK 22+ with Project Panama FFM)
```kotlin
dependencies {
    implementation("cc.projectargus:libargus-langchain4j:2.3.1")
    implementation("cc.projectargus:libargus-core:2.3.1")

    // Zero-config native binary extraction for your target OS/CPU/GPU:
    runtimeOnly("cc.projectargus:libargus-native-linux-cuda:2.3.1")
}
```

### Option B — JDK 17 / 21 LTS or Android (Standard JNI)
```kotlin
dependencies {
    implementation("cc.projectargus:libargus-langchain4j:2.3.1")
    implementation("cc.projectargus:libargus-jni:2.3.1")

    // Choose your platform runtime classifier (e.g., Linux CPU, macOS Metal, Windows Vulkan):
    runtimeOnly("cc.projectargus:libargus-native-linux-cpu:2.3.1")
}
```

---

## 2. Synchronous Chat & Per-Request Overrides (`LibArgusChatModel`)

`LibArgusChatModel` implements LangChain4j's `ChatModel` interface, supporting multi-turn conversation formatting (`LibArgusChatFormatter.autoDetect()`, `chatMl()`, `llama3()`, `gemma()`, `mistral()`, or custom `LibArgusChatFormatter`), per-request hyperparameter overrides (`ChatRequestParameters`), BPE-boundary-safe stop sequences, and exact token usage telemetry (`TokenUsage`, `FinishReason`).

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

## 3. Live Token Streaming & Cooperative Cancellation (`LibArgusStreamingChatModel`)

`LibArgusStreamingChatModel` implements `StreamingChatModel`, streaming decoded UTF-8 pieces to `StreamingChatResponseHandler.onPartialResponse` while holding back partial stop-sequence prefixes via `ArgusStopSequenceDetector` so stop markers never leak to the caller. Pass an `ArgusAbortFlag` to `doChat(...)` for lock-free cross-thread cancellation.

```java
// snippet-id: streaming-chat-abort
import cc.projectargus.libargus.ArgusAbortFlag;
import cc.projectargus.libargus.langchain4j.LibArgusStreamingChatModel;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class StreamingChatExample {
    public static void main(String[] args) {
        Path modelPath = Path.of(args.length > 0 ? args[0] : "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf");

        try (ArgusAbortFlag abortFlag = new ArgusAbortFlag();
             LibArgusStreamingChatModel streamModel = LibArgusStreamingChatModel.builder()
                .modelPath(modelPath)
                .contextLength(8192)
                .gpuLayers(0)
                .maxOutputTokens(256)
                .seed(42)
                .build()) {

            CompletableFuture<ChatResponse> done = new CompletableFuture<>();
            ChatRequest request = ChatRequest.builder()
                    .messages(List.of(UserMessage.from("Write a haiku about CPU cache lines.")))
                    .build();

            streamModel.doChat(request, new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String tokenPiece) {
                    System.out.print(tokenPiece);
                    System.out.flush();
                    // Call abortFlag.abort() from any thread (e.g. UI Stop button) to halt immediately
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
            }, abortFlag);

            done.join();
        }
    }
}
```

---

## 4. Bidirectional Document Embeddings for Local RAG (`LibArgusEmbeddingModel`)

`LibArgusEmbeddingModel` implements `EmbeddingModel` with L2-normalized vector extraction and cumulative `TokenUsage` tracking. Setting `.causalAttn(false)` preserves bidirectional non-causal attention masks required by BERT, `EmbeddingGemma 2`, and `jina-embeddings-v3`, and automatically sizes `uBatch` to `contextLength`.

```java
// snippet-id: rag-embeddings
import cc.projectargus.libargus.langchain4j.LibArgusEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
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
                .normalize(true)
                .build()) {

            List<TextSegment> docs = List.of(
                TextSegment.from("libargus unifies LLM, Whisper ASR, TTS, and Vision in one C++17 binary."),
                TextSegment.from("ArgusSequenceSlotPool partitions primary and worker clone KV cache slots.")
            );

            Response<List<Embedding>> result = embeddingModel.embedAll(docs);
            float[] v0 = result.content().get(0).vector();
            float[] v1 = result.content().get(1).vector();
            double dot = 0.0;
            for (int i = 0; i < v0.length; i++) {
                dot += (double) v0[i] * v1[i];
            }
            System.out.printf("Dimension: %d | Segments: %d | Prompt tokens: %d%n",
                    result.content().get(0).dimension(),
                    result.content().size(),
                    result.tokenUsage().inputTokenCount());
            System.out.printf("Cosine similarity (doc0 vs doc1): %.4f%n", dot);
        }
    }
}
```

---

## 5. Zero-Trust Multimodal Vision & Audio Chat (`ImageContent` & `AudioContent`)

When `.mmprojPath(...)` is configured on `LibArgusChatModel` or `LibArgusStreamingChatModel`, `UserMessage` instances containing `ImageContent` and `AudioContent` are automatically projected via `ArgusMultimodalContext` in-process.

> [!IMPORTANT]
> **Zero-Trust Media Sandbox Policy:**
> * **Inline Base64 (`ImageContent` / `AudioContent`):** Enabled by default and guarded by a pre-decode `.maxMediaBytes(...)` cap (default `64 MiB`) prior to Java heap allocation.
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

When passing an externally loaded `.model(ArgusModel)` to `LibArgusChatModel`, `LibArgusStreamingChatModel`, `LibArgusEmbeddingModel`, or `LibArgusTokenCountEstimator`, closing the LangChain4j wrapper closes only its dedicated `ArgusContext` while leaving the shared `ArgusModel` weights resident in memory.

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
