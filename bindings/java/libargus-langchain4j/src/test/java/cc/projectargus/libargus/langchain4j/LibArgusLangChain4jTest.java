package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusAbortFlag;
import cc.projectargus.libargus.ArgusBackend;
import cc.projectargus.libargus.ArgusModel;
import cc.projectargus.libargus.ArgusSamplerConfig;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.Response;
import java.io.File;
import java.io.FileDescriptor;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Fanged end-to-end integration, security, lifecycle, and dual-engine (JNI vs. Panama FFM)
 * verification suite for {@code :libargus-langchain4j}.
 */
public class LibArgusLangChain4jTest {

    private static final Set<String> SHARED_COMMON_CLASSES = Set.of(
        "cc.projectargus.libargus.ArgusLogLevel",
        "cc.projectargus.libargus.ArgusLogCallback",
        "cc.projectargus.libargus.ArgusPerfTimings",
        "cc.projectargus.libargus.ArgusSamplerConfig",
        "cc.projectargus.libargus.ArgusGenerationResult",
        "cc.projectargus.libargus.ArgusGenerationResult$StopReason",
        "cc.projectargus.libargus.ArgusNativeException",
        "cc.projectargus.libargus.ArgusNativeException$ErrorResolver",
        "cc.projectargus.libargus.internal.ArgusNativeLoader",
        "cc.projectargus.libargus.internal.AbstractNativeResource",
        "cc.projectargus.libargus.internal.AbstractNativeResource$HandleAction",
        "cc.projectargus.libargus.internal.ArgusUtf8StreamDecoder",
        "cc.projectargus.libargus.internal.ArgusStopSequenceDetector",
        "cc.projectargus.libargus.internal.ArgusCommonValidation",
        "cc.projectargus.libargus.internal.ArgusBackendSupport"
    );

    private static Path findProjectRoot() {
        Path root = Paths.get(System.getProperty("argus.root", "")).toAbsolutePath();
        while (root != null && !Files.exists(root.resolve("tests/data/tiny-embed.gguf"))) {
            root = root.getParent();
        }
        assertNotNull(root, "Could not locate project root containing tests/data/tiny-embed.gguf");
        return root;
    }

    private static Path findTinyModelPath() {
        return findEmbedModelPath();
    }

    private static Path findEmbedModelPath() {
        return findProjectRoot().resolve("tests/data/tiny-embed.gguf");
    }

    private static int extractPosixFd(FileDescriptor fdObj) throws Exception {
        Field fdField = FileDescriptor.class.getDeclaredField("fd");
        fdField.setAccessible(true);
        return fdField.getInt(fdObj);
    }

    @Test
    public void testChatAndStreamingChatEndToEndWithStopSequencesAndCancellation() throws Exception {
        ArgusBackend.init();
        Path modelPath = findTinyModelPath();

        try (LibArgusChatModel chatModel = LibArgusChatModel.builder()
                .modelPath(modelPath)
                .gpuLayers(0)
                .contextLength(128)
                .cpuThreads(2)
                .samplerConfig(ArgusSamplerConfig.greedy())
                .maxOutputTokens(6)
                .chatFormatter(LibArgusChatFormatter.fallbackInstruct())
                .build();
             LibArgusStreamingChatModel streamingModel = LibArgusStreamingChatModel.builder()
                .model(chatModel.getModel())
                .contextLength(128)
                .cpuThreads(2)
                .samplerConfig(ArgusSamplerConfig.greedy())
                .maxOutputTokens(6)
                .chatFormatter(LibArgusChatFormatter.fallbackInstruct())
                .build()) {

            // 1. Synchronous chat(String) and chat(ChatRequest)
            String simpleReply = chatModel.chat("Hello world");
            assertNotNull(simpleReply);
            assertFalse(simpleReply.isEmpty(), "Generated reply must be non-empty");

            ChatRequest req = ChatRequest.builder()
                .messages(UserMessage.from("Hello world"))
                .build();
            ChatResponse syncResp = chatModel.chat(req);
            assertNotNull(syncResp);
            assertEquals(simpleReply, syncResp.aiMessage().text());
            assertNotNull(syncResp.tokenUsage());
            assertTrue(syncResp.tokenUsage().inputTokenCount() > 0);
            assertEquals(6, syncResp.tokenUsage().outputTokenCount());
            assertEquals(FinishReason.LENGTH, syncResp.finishReason());

            // 2. StreamingChatModel parity: concatenation of onPartialResponse pieces == full text
            List<String> streamedPieces = new ArrayList<>();
            AtomicReference<ChatResponse> completedResp = new AtomicReference<>();
            AtomicReference<Throwable> streamError = new AtomicReference<>();
            CountDownLatch latch = new CountDownLatch(1);

            streamingModel.chat(req, new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String partialResponse) {
                    streamedPieces.add(partialResponse);
                }

                @Override
                public void onCompleteResponse(ChatResponse completeResponse) {
                    completedResp.set(completeResponse);
                    latch.countDown();
                }

                @Override
                public void onError(Throwable error) {
                    streamError.set(error);
                    latch.countDown();
                }
            });

            assertTrue(latch.await(5, TimeUnit.SECONDS));
            assertNull(streamError.get());
            assertNotNull(completedResp.get());
            assertEquals(simpleReply, completedResp.get().aiMessage().text());
            assertEquals(simpleReply, String.join("", streamedPieces));
            assertEquals(syncResp.tokenUsage(), completedResp.get().tokenUsage());
            assertEquals(FinishReason.LENGTH, completedResp.get().finishReason());

            // 3. Stop-sequence early halting & zero-leak streaming holdback verification
            assertTrue(simpleReply.length() >= 2, "Expected >= 2 chars in greedy output to test stop sequence");
            int splitPoint = Math.max(1, simpleReply.length() / 2);
            String expectedPrefix = simpleReply.substring(0, splitPoint);
            String stopSeq = simpleReply.substring(splitPoint);

            ChatRequest stopReq = ChatRequest.builder()
                .messages(UserMessage.from("Hello world"))
                .parameters(ChatRequestParameters.builder()
                    .stopSequences(List.of(stopSeq))
                    .maxOutputTokens(6)
                    .build())
                .build();

            ChatResponse syncStopped = chatModel.chat(stopReq);
            assertEquals(expectedPrefix, syncStopped.aiMessage().text());
            assertFalse(syncStopped.aiMessage().text().contains(stopSeq));
            assertEquals(FinishReason.STOP, syncStopped.finishReason());

            List<String> stopStreamedPieces = new ArrayList<>();
            AtomicReference<ChatResponse> stopStreamResp = new AtomicReference<>();
            streamingModel.chat(stopReq, new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String partialResponse) {
                    stopStreamedPieces.add(partialResponse);
                }

                @Override
                public void onCompleteResponse(ChatResponse completeResponse) {
                    stopStreamResp.set(completeResponse);
                }

                @Override
                public void onError(Throwable error) {
                    fail("Unexpected streaming error", error);
                }
            });

            assertNotNull(stopStreamResp.get());
            assertEquals(expectedPrefix, stopStreamResp.get().aiMessage().text());
            assertEquals(expectedPrefix, String.join("", stopStreamedPieces),
                "Stop sequence must never leak into streamed onPartialResponse chunks");
            assertEquals(FinishReason.STOP, stopStreamResp.get().finishReason());

            // 4. Cooperative mid-stream cancellation via ArgusAbortFlag
            try (ArgusAbortFlag abortFlag = new ArgusAbortFlag()) {
                List<String> abortedPieces = new ArrayList<>();
                AtomicReference<ChatResponse> abortedResp = new AtomicReference<>();
                streamingModel.doChat(req, new StreamingChatResponseHandler() {
                    @Override
                    public void onPartialResponse(String partialResponse) {
                        abortedPieces.add(partialResponse);
                        abortFlag.abort(); // Cancel immediately after first emitted piece!
                    }

                    @Override
                    public void onCompleteResponse(ChatResponse completeResponse) {
                        abortedResp.set(completeResponse);
                    }

                    @Override
                    public void onError(Throwable error) {
                        fail("Abort should complete cleanly with FinishReason.OTHER, not error", error);
                    }
                }, abortFlag);

                assertNotNull(abortedResp.get());
                assertEquals(FinishReason.OTHER, abortedResp.get().finishReason());
                assertEquals(1, abortedPieces.size(), "Generation must halt immediately after abortFlag.abort()");
                assertTrue(abortedResp.get().tokenUsage().outputTokenCount() < 6);
            }
        }
    }

    @Test
    public void testEmbeddingModelAndTokenCountEstimator() {
        ArgusBackend.init();
        Path embedModelPath = findEmbedModelPath();

        try (LibArgusEmbeddingModel embeddingModel = LibArgusEmbeddingModel.builder()
                .modelPath(embedModelPath)
                .gpuLayers(0)
                .contextLength(128)
                .cpuThreads(2)
                .normalize(true)
                .queryPrefix("search_query: ")
                .documentPrefix("search_document: ")
                .build();
             LibArgusTokenCountEstimator estimator = LibArgusTokenCountEstimator.fromModel(embeddingModel.getModel())) {

            int dim = embeddingModel.dimension();
            assertEquals(embeddingModel.getModel().nEmbd(), dim);
            assertFalse(embeddingModel.getContext().isCausalAttn(),
                "LibArgusEmbeddingModel must default to bidirectional non-causal attention");

            // 1. Single query embedding via embed(String)
            Response<Embedding> queryResp = embeddingModel.embed("What is mechanical sympathy?");
            assertNotNull(queryResp.content());
            assertEquals(dim, queryResp.content().dimension());
            assertTrue(queryResp.tokenUsage().inputTokenCount() > 0);
            assertEquals(0, queryResp.tokenUsage().outputTokenCount());

            // Verify unit L2 norm
            float[] vec = queryResp.content().vector();
            double normSq = 0.0;
            for (float v : vec) {
                assertTrue(Float.isFinite(v), "Embedding component must be finite");
                normSq += (double) v * (double) v;
            }
            assertEquals(1.0, Math.sqrt(normSq), 1e-4, "Normalized embedding must have unit L2 norm");

            // 2. Batch embedAll with document vs query metadata prefixing
            TextSegment docSeg = TextSegment.from("What is mechanical sympathy?");
            TextSegment querySeg = TextSegment.from(
                "What is mechanical sympathy?",
                Metadata.from(LibArgusEmbeddingModel.METADATA_KEY_EMBEDDING_TYPE, "query")
            );
            Response<List<Embedding>> batchResp = embeddingModel.embedAll(List.of(docSeg, querySeg));
            assertEquals(2, batchResp.content().size());
            assertArrayEquals(vec, batchResp.content().get(1).vector(), 1e-5f,
                "Query-tagged segment must match embed(String) output");
            assertFalse(Arrays.equals(batchResp.content().get(0).vector(), batchResp.content().get(1).vector()),
                "Document prefix and query prefix must produce distinct embeddings");

            int expectedDocTokens = estimator.estimateTokenCountInText("search_document: What is mechanical sympathy?") + 1;
            int expectedQueryTokens = estimator.estimateTokenCountInText("search_query: What is mechanical sympathy?") + 1;
            assertTrue(batchResp.tokenUsage().inputTokenCount() > 0);
            assertTrue(expectedDocTokens > 0 && expectedQueryTokens > 0);

            // 3. TokenCountEstimator edge cases (empty string, ASCII, CJK, 4-byte emoji, ChatMessages)
            assertEquals(0, estimator.estimateTokenCountInText(""));
            assertTrue(estimator.estimateTokenCountInText("Hello world") > 0);
            assertTrue(estimator.estimateTokenCountInText("你好世界 🚀🔥🧠") > 0);
            assertTrue(estimator.estimateTokenCountInMessage(UserMessage.from("Count this message")) > 0);
            assertTrue(estimator.estimateTokenCountInMessages(List.of(
                SystemMessage.from("System prompt"),
                UserMessage.from("User prompt")
            )) > 0);
            assertEquals(0, estimator.estimateTokenCountInMessages(List.of()));
        }
    }

    @Test
    public void testSharedModelOwnershipAndFdLoadingLifecycle() throws Exception {
        ArgusBackend.init();
        Path modelPath = findTinyModelPath();

        // 1. Shared ArgusModel across multiple adapters: closing adapters must NOT close caller's ArgusModel
        try (ArgusModel sharedModel = ArgusModel.load(modelPath, 0, false)) {
            LibArgusChatModel chat = LibArgusChatModel.builder()
                .model(sharedModel)
                .contextLength(128)
                .cpuThreads(2)
                .maxOutputTokens(2)
                .build();
            LibArgusStreamingChatModel stream = LibArgusStreamingChatModel.builder()
                .model(sharedModel)
                .contextLength(128)
                .cpuThreads(2)
                .maxOutputTokens(2)
                .build();
            LibArgusTokenCountEstimator estimator = LibArgusTokenCountEstimator.fromModel(sharedModel);

            assertNotNull(chat.chat("Test 1"));
            chat.close();
            assertTrue(chat.isClosed());
            assertThrows(IllegalStateException.class, () -> chat.chat("After close"));

            // Sibling adapters and sharedModel must remain 100% operational!
            assertFalse(sharedModel.isClosed());
            assertFalse(stream.isClosed());
            assertFalse(estimator.isClosed());
            assertTrue(estimator.estimateTokenCountInText("Still alive") > 0);

            AtomicReference<ChatResponse> resp = new AtomicReference<>();
            stream.chat(ChatRequest.builder().messages(UserMessage.from("Still alive")).build(),
                new StreamingChatResponseHandler() {
                    @Override public void onPartialResponse(String partialResponse) {}
                    @Override public void onCompleteResponse(ChatResponse completeResponse) { resp.set(completeResponse); }
                    @Override public void onError(Throwable error) { fail(error); }
                });
            assertNotNull(resp.get());

            stream.close();
            estimator.close();
            assertFalse(sharedModel.isClosed(), "Closing non-owning adapters must never close shared ArgusModel");
        }

        // 2. Standalone modelFd loading: adapter owns model and closes it on close()
        try (RandomAccessFile raf = new RandomAccessFile(modelPath.toFile(), "r")) {
            int fd = extractPosixFd(raf.getFD());
            ArgusModel ownedRef;
            try (LibArgusChatModel fdChat = LibArgusChatModel.builder()
                    .modelFd(fd)
                    .gpuLayers(0)
                    .contextLength(128)
                    .cpuThreads(2)
                    .maxOutputTokens(2)
                    .build()) {
                ownedRef = fdChat.getModel();
                assertFalse(ownedRef.isClosed());
                assertNotNull(fdChat.chat("FD loaded model"));
            }
            assertTrue(ownedRef.isClosed(), "Standalone modelFd adapter must close its owned ArgusModel");
        }
    }

    @Test
    public void testZeroTrustMediaSecurityPolicy(@TempDir Path tempDir) throws Exception {
        Path allowedDir = tempDir.resolve("allowed");
        Path outsideDir = tempDir.resolve("outside");
        Files.createDirectories(allowedDir);
        Files.createDirectories(outsideDir);
        Path secretFile = outsideDir.resolve("secret.bin");
        Files.write(secretFile, new byte[] {1, 2, 3, 4});
        Path insideFile = allowedDir.resolve("inside.bin");
        Files.write(insideFile, new byte[] {1, 2, 3, 4});

        // 1. Remote HTTP/HTTPS URIs must be unconditionally rejected (SSRF protection)
        UserMessage ssrfImageMsg = UserMessage.from(
            TextContent.from("Describe this"),
            ImageContent.from(URI.create("http://169.254.169.254/latest/meta-data/"))
        );
        IllegalArgumentException ssrfEx = assertThrows(IllegalArgumentException.class, () ->
            LibArgusMediaSupport.extractBitmaps(null, List.of(ssrfImageMsg), allowedDir, true, 1024)
        );
        assertTrue(ssrfEx.getMessage().contains("SSRF"));

        // 2. Local file:// URIs must be rejected by default when neither allowedMediaRoot nor allowLocalFileMedia is set
        UserMessage localFileMsg = UserMessage.from(
            TextContent.from("Describe file"),
            ImageContent.from(insideFile.toUri())
        );
        IllegalArgumentException disabledEx = assertThrows(IllegalArgumentException.class, () ->
            LibArgusMediaSupport.extractBitmaps(null, List.of(localFileMsg), null, false, 1024)
        );
        assertTrue(disabledEx.getMessage().contains("allowedMediaRoot"));

        // 3. Path escaping allowedMediaRoot must throw SecurityException before touching native layer
        UserMessage escapeMsg = UserMessage.from(
            TextContent.from("Read outside file"),
            AudioContent.from(secretFile.toUri())
        );
        assertThrows(SecurityException.class, () ->
            LibArgusMediaSupport.extractBitmaps(null, List.of(escapeMsg), allowedDir, false, 1024)
        );

        // 3b. Non-regular file (e.g. directory) inside allowedMediaRoot must be rejected before native load
        UserMessage dirMsg = UserMessage.from(
            TextContent.from("Read directory"),
            ImageContent.from(allowedDir.toUri())
        );
        IllegalArgumentException dirEx = assertThrows(IllegalArgumentException.class, () ->
            LibArgusMediaSupport.extractBitmaps(null, List.of(dirMsg), allowedDir, false, 1024)
        );
        assertTrue(dirEx.getMessage().contains("regular file"));

        // 4. Oversized Base64 payload must be rejected before decoding (OOM DoS defense)
        String oversizedBase64 = Base64.getEncoder().encodeToString(new byte[256]);
        UserMessage bombMsg = UserMessage.from(
            TextContent.from("Decode bomb"),
            ImageContent.from(oversizedBase64, "image/png")
        );
        IllegalArgumentException sizeEx = assertThrows(IllegalArgumentException.class, () ->
            LibArgusMediaSupport.extractBitmaps(null, List.of(bombMsg), null, false, 64)
        );
        assertTrue(sizeEx.getMessage().contains("maxMediaBytes"));

        // 5. Sending media to a LibArgusChatModel without mmprojPath must fail fast with IllegalStateException
        try (LibArgusChatModel textOnlyModel = LibArgusChatModel.builder()
                .modelPath(findTinyModelPath())
                .gpuLayers(0)
                .contextLength(128)
                .cpuThreads(2)
                .build()) {
            ChatRequest mediaReq = ChatRequest.builder().messages(bombMsg).build();
            assertThrows(IllegalStateException.class, () -> textOnlyModel.chat(mediaReq));
        }
    }

    @Test
    public void testChatFormatterTemplatesAndMediaMarkerInjection() {
        List<ChatMessage> conversation = List.of(
            SystemMessage.from("You are a helpful systems architect."),
            UserMessage.from(
                TextContent.from("Inspect this diagram:"),
                ImageContent.from("AAAA", "image/png")
            ),
            AiMessage.from(ToolExecutionRequest.builder().id("1").name("lookup").arguments("{\"k\":1}").build()),
            ToolExecutionResultMessage.from("1", "lookup", "ok")
        );

        String chatMl = LibArgusChatFormatter.chatMl().format(null, conversation, true);
        assertTrue(chatMl.contains("<|im_start|>system\nYou are a helpful systems architect.<|im_end|>"));
        assertTrue(chatMl.contains("<__media__>\n"));
        assertTrue(chatMl.contains("[ToolCall: lookup({\"k\":1})]"));
        assertTrue(chatMl.contains("<|im_start|>tool\n[lookup] ok<|im_end|>"));
        assertTrue(chatMl.endsWith("<|im_start|>assistant\n"));

        String llama3 = LibArgusChatFormatter.llama3().format(null, conversation, false);
        assertTrue(llama3.contains("<|start_header_id|>system<|end_header_id|>"));
        assertFalse(llama3.contains("<__media__>"));
        assertTrue(llama3.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        String gemma = LibArgusChatFormatter.gemma().format(null, conversation, true);
        assertTrue(gemma.contains("<start_of_turn>user\nYou are a helpful systems architect.\n\nInspect this diagram:\n<__media__>\n<end_of_turn>"));
        assertTrue(gemma.endsWith("<start_of_turn>model\n"));

        String mistral = LibArgusChatFormatter.mistral().format(null, conversation, true);
        assertTrue(mistral.contains("[INST] You are a helpful systems architect.\n\nInspect this diagram:\n<__media__>\n [/INST]"));
    }

    @Test
    public void testDualEngineJniAndPanamaFfmEquivalence() throws Exception {
        ArgusBackend.init();
        Path modelPath = findEmbedModelPath();

        String coreJarPath = System.getProperty("argus.core.jar");
        assertNotNull(coreJarPath, "System property argus.core.jar must be set by Gradle test task");
        File coreJar = new File(coreJarPath);
        assertTrue(coreJar.isFile(), "libargus-core jar does not exist at: " + coreJarPath);

        URL lc4jClassesUrl = LibArgusChatModel.class.getProtectionDomain().getCodeSource().getLocation();
        URL[] urls = new URL[] { coreJar.toURI().toURL(), lc4jClassesUrl };
        ClassLoader parent = LibArgusLangChain4jTest.class.getClassLoader();

        try (URLClassLoader ffmLoader = new URLClassLoader(urls, parent) {
            @Override
            protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
                synchronized (getClassLoadingLock(name)) {
                    if (name.startsWith("cc.projectargus.libargus.")
                        && !name.startsWith("cc.projectargus.libargus.spi.")
                        && !SHARED_COMMON_CLASSES.contains(name)) {
                        Class<?> c = findLoadedClass(name);
                        if (c == null) {
                            try {
                                c = findClass(name);
                            } catch (ClassNotFoundException ignored) {
                                return super.loadClass(name, resolve);
                            }
                        }
                        if (resolve) {
                            resolveClass(c);
                        }
                        return c;
                    }
                    return super.loadClass(name, resolve);
                }
            }
        }) {
            // Verify that ffmLoader loaded ArgusModel from :libargus-core (which extends ArgusNativeResource)
            Class<?> ffmArgusModelCls = ffmLoader.loadClass("cc.projectargus.libargus.ArgusModel");
            assertEquals(
                "cc.projectargus.libargus.internal.ArgusNativeResource",
                ffmArgusModelCls.getSuperclass().getName(),
                "Isolated classloader must bind LibArgus* to :libargus-core (Panama FFM)"
            );

            // 1. Compare LibArgusChatModel over JNI vs. Panama FFM
            Class<?> ffmChatModelCls = ffmLoader.loadClass("cc.projectargus.libargus.langchain4j.LibArgusChatModel");
            Object ffmChatBuilder = ffmChatModelCls.getMethod("builder").invoke(null);
            Class<?> ffmChatBuilderCls = ffmChatBuilder.getClass();
            ffmChatBuilderCls.getMethod("modelPath", Path.class).invoke(ffmChatBuilder, modelPath);
            ffmChatBuilderCls.getMethod("gpuLayers", int.class).invoke(ffmChatBuilder, 0);
            ffmChatBuilderCls.getMethod("contextLength", int.class).invoke(ffmChatBuilder, 128);
            ffmChatBuilderCls.getMethod("cpuThreads", int.class).invoke(ffmChatBuilder, 2);
            ffmChatBuilderCls.getMethod("samplerConfig", ArgusSamplerConfig.class).invoke(ffmChatBuilder, ArgusSamplerConfig.greedy());
            ffmChatBuilderCls.getMethod("maxOutputTokens", int.class).invoke(ffmChatBuilder, 4);

            ChatRequest chatReq = ChatRequest.builder().messages(UserMessage.from("Parity check")).build();
            try (AutoCloseable ffmChat = (AutoCloseable) ffmChatBuilderCls.getMethod("build").invoke(ffmChatBuilder);
                 LibArgusChatModel jniChat = LibArgusChatModel.builder()
                    .modelPath(modelPath)
                    .gpuLayers(0)
                    .contextLength(128)
                    .cpuThreads(2)
                    .samplerConfig(ArgusSamplerConfig.greedy())
                    .maxOutputTokens(4)
                    .build()) {

                Method ffmDoChat = ffmChatModelCls.getMethod("chat", ChatRequest.class);
                ChatResponse ffmResp = (ChatResponse) ffmDoChat.invoke(ffmChat, chatReq);
                ChatResponse jniResp = jniChat.chat(chatReq);

                assertEquals(jniResp.aiMessage().text(), ffmResp.aiMessage().text(),
                    "LibArgusChatModel must produce bit-identical text on Panama FFM and JNI");
                assertEquals(jniResp.tokenUsage(), ffmResp.tokenUsage(),
                    "LibArgusChatModel must report identical TokenUsage on Panama FFM and JNI");
                assertEquals(jniResp.finishReason(), ffmResp.finishReason());
            }

            // 2. Compare LibArgusEmbeddingModel over JNI vs. Panama FFM
            Class<?> ffmEmbedModelCls = ffmLoader.loadClass("cc.projectargus.libargus.langchain4j.LibArgusEmbeddingModel");
            Object ffmEmbedBuilder = ffmEmbedModelCls.getMethod("builder").invoke(null);
            Class<?> ffmEmbedBuilderCls = ffmEmbedBuilder.getClass();
            ffmEmbedBuilderCls.getMethod("modelPath", Path.class).invoke(ffmEmbedBuilder, modelPath);
            ffmEmbedBuilderCls.getMethod("gpuLayers", int.class).invoke(ffmEmbedBuilder, 0);
            ffmEmbedBuilderCls.getMethod("contextLength", int.class).invoke(ffmEmbedBuilder, 128);
            ffmEmbedBuilderCls.getMethod("cpuThreads", int.class).invoke(ffmEmbedBuilder, 2);

            try (AutoCloseable ffmEmbed = (AutoCloseable) ffmEmbedBuilderCls.getMethod("build").invoke(ffmEmbedBuilder);
                 LibArgusEmbeddingModel jniEmbed = LibArgusEmbeddingModel.builder()
                    .modelPath(modelPath)
                    .gpuLayers(0)
                    .contextLength(128)
                    .cpuThreads(2)
                    .build()) {

                Method ffmEmbedText = ffmEmbedModelCls.getMethod("embed", String.class);
                @SuppressWarnings("unchecked")
                Response<Embedding> ffmEmbResp = (Response<Embedding>) ffmEmbedText.invoke(ffmEmbed, "Dual-engine vector parity");
                Response<Embedding> jniEmbResp = jniEmbed.embed("Dual-engine vector parity");

                assertArrayEquals(jniEmbResp.content().vector(), ffmEmbResp.content().vector(), 1e-6f,
                    "LibArgusEmbeddingModel must produce bit-identical embeddings across Panama FFM and JNI");
                assertEquals(jniEmbResp.tokenUsage(), ffmEmbResp.tokenUsage());
            }
        }
    }
}
