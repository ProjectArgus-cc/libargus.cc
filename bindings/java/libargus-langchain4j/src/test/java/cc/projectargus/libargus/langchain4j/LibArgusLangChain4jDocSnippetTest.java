package cc.projectargus.libargus.langchain4j;

import static org.junit.jupiter.api.Assertions.*;

import cc.projectargus.libargus.ArgusContextConfig;
import cc.projectargus.libargus.testutil.ArgusDocSnippetRunner;
import cc.projectargus.libargus.testutil.ArgusDocSnippetRunner.JavaSnippet;
import cc.projectargus.libargus.testutil.ArgusSyntheticTestFixtures;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Executable Markdown documentation doctest suite for {@code docs/quickstart-langchain4j.md}
 * and the LangChain4j section of {@code docs/quickstart-java.md}.
 */
class LibArgusLangChain4jDocSnippetTest {

    @TempDir
    Path tempDir;

    @Test
    void testQuickstartLangChain4jAllFiveRecipesCompileAndExecute() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path lang4jMd = root.resolve("docs/quickstart-langchain4j.md");
        List<JavaSnippet> snippets = ArgusDocSnippetRunner.extractJavaSnippets(lang4jMd);
        assertEquals(5, snippets.size(), "Expected 5 Java recipes in docs/quickstart-langchain4j.md");

        Path tinyGguf = ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir);
        Path tinyEmbedGguf = root.resolve("tests/data/tiny-embed.gguf");
        Path omniMmprojGguf = ArgusSyntheticTestFixtures.generateTinyOmniProjector(tempDir);
        Path samplePng = ArgusSyntheticTestFixtures.generateTestPng(tempDir, "diagram.png");
        Path sampleWav = ArgusSyntheticTestFixtures.generateTestWav16kHz(tempDir, "sample.wav");

        // Recipe 0: SyncChatExample
        String syncOut = ArgusDocSnippetRunner.compileAndRunMain(
                snippets.get(0),
                src -> src
                        .replace(".contextLength(8192)", ".contextLength(512)")
                        .replace(".maxOutputTokens(256)", ".maxOutputTokens(4)")
                        .replace(".maxOutputTokens(16)", ".maxOutputTokens(4)"),
                tinyGguf.toString());
        assertTrue(syncOut.contains("Assistant:"), "Expected Assistant output from SyncChatExample, got: " + syncOut);
        assertTrue(syncOut.contains("Tokens (in/out/total):"),
                "Expected token telemetry from SyncChatExample, got: " + syncOut);

        // Recipe 1: StreamingChatExample
        String streamOut = ArgusDocSnippetRunner.compileAndRunMain(
                snippets.get(1),
                src -> src
                        .replace(".contextLength(8192)", ".contextLength(512)")
                        .replace(".maxOutputTokens(256)", ".maxOutputTokens(4)"),
                tinyGguf.toString());
        assertTrue(streamOut.contains("[Finished:"),
                "Expected finish reason from StreamingChatExample, got: " + streamOut);

        // Recipe 2: EmbeddingRagExample
        String embedOut = ArgusDocSnippetRunner.compileAndRunMain(
                snippets.get(2),
                src -> src.replace(".contextLength(512)", ".contextLength(128)"),
                tinyEmbedGguf.toString());
        assertTrue(embedOut.contains("Dimension: 32"),
                "Expected 32-d vector from EmbeddingRagExample, got: " + embedOut);
        assertTrue(embedOut.contains("Cosine similarity (doc0 vs doc1):"),
                "Expected cosine similarity output from EmbeddingRagExample, got: " + embedOut);

        // Recipe 3: MultimodalChatExample (Vision PNG + Audio WAV via unified tiny-mmproj-omni.gguf)
        String mmOut = ArgusDocSnippetRunner.compileAndRunMain(
                snippets.get(3),
                src -> src
                        .replace(".contextLength(8192)", ".contextLength(512)")
                        .replace(".maxOutputTokens(256)", ".maxOutputTokens(4)"),
                tinyGguf.toString(),
                omniMmprojGguf.toString(),
                tempDir.toString(),
                samplePng.toString(),
                sampleWav.toString());
        assertFalse(mmOut.isBlank(), "Expected non-empty response from MultimodalChatExample");

        // Recipe 4: SharedModelExample
        String sharedOut = ArgusDocSnippetRunner.compileAndRunMain(
                snippets.get(4),
                src -> src
                        .replace(".contextLength(4096)", ".contextLength(512)")
                        .replace(".maxOutputTokens(128)", ".maxOutputTokens(4)"),
                tinyGguf.toString());
        assertTrue(sharedOut.contains("Prompt token count:"),
                "Expected prompt token count from SharedModelExample, got: " + sharedOut);
        assertTrue(sharedOut.contains("Fast reply:"),
                "Expected Fast reply from SharedModelExample, got: " + sharedOut);
        assertTrue(sharedOut.contains("Creative reply:"),
                "Expected Creative reply from SharedModelExample, got: " + sharedOut);
    }

    @Test
    void testQuickstartJavaSection9LangChain4jSnippetCompilesAndExecutes() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path quickstartJava = root.resolve("docs/quickstart-java.md");
        List<JavaSnippet> snippets = ArgusDocSnippetRunner.extractJavaSnippets(quickstartJava);
        assertEquals(14, snippets.size());

        JavaSnippet lang4jTeaser = snippets.get(13);
        String tinyGguf = ArgusDocSnippetRunner.escapeJavaPath(
                ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir));

        String out = ArgusDocSnippetRunner.compileAndRunMain(
                lang4jTeaser,
                src -> src
                        .replace("models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", tinyGguf)
                        .replace(".contextLength(8192)", ".contextLength(64)")
                        .replace(".maxOutputTokens(512)", ".maxOutputTokens(4)"));
        assertFalse(out.isBlank(), "Expected non-empty output from Section 9 LangChain4j snippet");
    }

    @Test
    void testChainedModelToModelPipeline() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path tinyGguf = ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir);
        Path tinyEmbedGguf = root.resolve("tests/data/tiny-embed.gguf");

        try (LibArgusChatModel chatModel = LibArgusChatModel.builder()
                .modelPath(tinyGguf)
                .contextLength(64)
                .maxOutputTokens(6)
                .temperature(0.0f)
                .seed(42)
                .build();
             LibArgusTokenCountEstimator estimator = LibArgusTokenCountEstimator.builder()
                .model(chatModel.getModel())
                .build();
             LibArgusEmbeddingModel embeddingModel = LibArgusEmbeddingModel.builder()
                .modelPath(tinyEmbedGguf)
                .contextConfig(new ArgusContextConfig.Builder(64)
                        .embeddings(true)
                        .causalAttn(false)
                        .build())
                .normalize(true)
                .build()) {

            // 1. Generate response from ChatModel
            ChatResponse chatResponse = chatModel.chat(
                    SystemMessage.from("You are a terse systems architect."),
                    UserMessage.from("Summarize off-heap memory."));
            String generatedText = chatResponse.aiMessage().text();
            assertNotNull(generatedText);
            assertFalse(generatedText.isEmpty());

            // 2. Estimate tokens on the generated AiMessage
            int estimatedTokens = estimator.estimateTokenCountInMessage(chatResponse.aiMessage());
            assertTrue(estimatedTokens > 0, "Expected positive token count for generated AiMessage");

            // 3. Feed generated text directly into EmbeddingModel
            Embedding embedding = embeddingModel.embed(generatedText).content();
            assertEquals(32, embedding.dimension());
            double normSq = 0.0;
            for (float v : embedding.vector()) {
                assertTrue(Float.isFinite(v));
                normSq += (double) v * v;
            }
            assertEquals(1.0, Math.sqrt(normSq), 1e-4, "Chained embedding output must be L2-normalized");
        }
    }
}
