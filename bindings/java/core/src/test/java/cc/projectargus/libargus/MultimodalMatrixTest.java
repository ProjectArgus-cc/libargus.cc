package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.io.File;

/**
 * Conditional integration testing framework verifying the active direct-inference cells
 * of the Multimodal Input/Output matrix.
 */
public class MultimodalMatrixTest {

    private static final Path ROOT = findProjectRoot();
    private static final Path MODELS_DIR = ROOT.resolve("models");
    private static final Path MEDIA_DIR = ROOT.resolve("media");

    // VLM Suite (Cells 1, 3, 5)
    private static final Path VLM_BASE = MODELS_DIR.resolve("qwen2-vl-2b-it-Q4_K_M.gguf");
    private static final Path VLM_PROJ = MODELS_DIR.resolve("qwen2-vl-2b-it-mmproj.gguf");

    // TTS Suite (Cell 2)
    private static final Path TTS_BASE = MODELS_DIR.resolve("OuteTTS-0.2-500M-Q4_K_M.gguf");
    private static final Path TTS_VOC = MODELS_DIR.resolve("wavtokenizer-large-Q4_K_M.gguf");

    // ASR Suite (Cell 7)
    private static final Path ASR_MODEL = MODELS_DIR.resolve("whisper-base.gguf");

    // Optional video media file
    private static final Path TEST_VIDEO = MEDIA_DIR.resolve("video.mp4");

    private static Path findProjectRoot() {
        Path path = Paths.get("").toAbsolutePath();
        while (path != null && !Files.exists(path.resolve("settings.gradle.kts"))) {
            path = path.getParent();
        }
        return path != null ? path : Paths.get("").toAbsolutePath();
    }

    @BeforeAll
    public static void setUp() {
        System.out.println("[MultimodalMatrixTest] Initializing ArgusBackend...");
        boolean initStatus = ArgusBackend.init();
        assertTrue(initStatus, "Global backend initialization failed");
    }

    @AfterAll
    public static void tearDown() {
        System.out.println("[MultimodalMatrixTest] Freeing ArgusBackend...");
        ArgusBackend.free();
    }

    @Test
    public void testCell1_TextToText() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell 1: Text In -> Text Out...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing at: " + VLM_BASE);

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, VLM_BASE, 99, true)) {
            
            ArgusContextConfig config = new ArgusContextConfig(null, 1024, 4, 0, 0, 0, false);

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                String prompt = "<|im_start|>user\nTell me a short one-sentence joke.<|im_end|>\n<|im_start|>assistant\n";
                MemorySegment promptSeg = arena.allocateFrom(prompt);
                
                // Tokenize prompt
                int maxTokens = 1024;
                MemorySegment tokenBuf = arena.allocate(ValueLayout.JAVA_INT, maxTokens);
                int nTokens = context.tokenize(promptSeg, tokenBuf, true);
                assertTrue(nTokens > 0);

                // Evaluate prompt batch
                int res = context.decodeBatch(tokenBuf, nTokens, 0, 0, true);
                assertEquals(0, res);

                // Autoregressive sampling loop
                StringBuilder response = new StringBuilder();
                int currentPos = nTokens;
                int token = context.sampleToken(0, 0.7f, 1.1f);
                int limit = 50;

                while (limit-- > 0) {
                    // Check standard End-Of-Generation token IDs
                    if (token == 151643 || token == 151645 || token == 128009 || token == 128001 || token == 2) {
                        break;
                    }
                    
                    String piece = context.tokenToPiece(token);
                    if (piece != null && !piece.isEmpty()) {
                        response.append(piece);
                    }

                    // Decode next token
                    MemorySegment nextTokenBuf = arena.allocate(ValueLayout.JAVA_INT, 1);
                    nextTokenBuf.setAtIndex(ValueLayout.JAVA_INT, 0, token);
                    int decodeRes = context.decodeBatch(nextTokenBuf, 1, currentPos, 0, true);
                    if (decodeRes != 0) {
                        break;
                    }
                    currentPos++;
                    token = context.sampleToken(0, 0.7f, 1.1f);
                }

                System.out.println("  - Joke generated: " + response.toString().trim());
                assertFalse(response.toString().trim().isEmpty(), "Generated response was empty");
            }
        }
    }

    @Test
    public void testCell2_TextToAudio() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell 2: Text In -> Audio Out...");
        Assumptions.assumeTrue(Files.exists(TTS_BASE), "Skipping: TTS Base model missing at: " + TTS_BASE);
        Assumptions.assumeTrue(Files.exists(TTS_VOC), "Skipping: TTS Vocoder model missing at: " + TTS_VOC);

        try (Arena arena = Arena.ofConfined();
             ArgusModel baseModel = ArgusModel.load(arena, TTS_BASE, 99, true);
             ArgusModel vocoderModel = ArgusModel.load(arena, TTS_VOC, 99, true)) {

            ArgusContextConfig config = new ArgusContextConfig(null, 8192, 4, 0, 0, 0, false);

            try (ArgusContext context = ArgusContext.init(arena, baseModel, config)) {
                String phrase = "Hello, this is a validation of the unmanaged Project Panama speech synthesis system.";
                MemorySegment textSeg = arena.allocateFrom(phrase);
                int maxSamples = 24000 * 10; // 10 seconds capacity
                MemorySegment outPcmSeg = arena.allocate(ValueLayout.JAVA_FLOAT, maxSamples);

                int samplesGenerated = context.synthesizeSpeech(vocoderModel, textSeg, 42, outPcmSeg, maxSamples);
                System.out.println("  - samplesGenerated returned: " + samplesGenerated);
                assertTrue(samplesGenerated > 0, "TTS Speech synthesis failed to generate samples: " + samplesGenerated);
                System.out.println("  - Generated " + samplesGenerated + " audio samples.");

                // Write generated PCM to test outputs directory
                File testOutputFile = ROOT.resolve("bindings/java/build/test-outputs/cell2_tts.wav").toFile();
                WavWriter.writeWav(outPcmSeg, samplesGenerated, 24000.0f, testOutputFile);
                assertTrue(testOutputFile.exists());
                System.out.println("  - Saved audio to: " + testOutputFile.getAbsolutePath());
            }
        } catch (Exception e) {
            fail("Speech synthesis exception occurred", e);
        }
    }

    @Test
    public void testCell3_ImageToText() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell 3: Image In -> Text Out...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing at: " + VLM_BASE);
        Assumptions.assumeTrue(Files.exists(VLM_PROJ), "Skipping: VLM Projector missing at: " + VLM_PROJ);

        try (Arena arena = Arena.ofConfined();
             ArgusModel baseModel = ArgusModel.load(arena, VLM_BASE, 99, true);
             ArgusContext context = ArgusContext.init(arena, baseModel, ArgusContextConfig.createDefault(8192));
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(arena, baseModel, VLM_PROJ, 4, true)) {

            assertTrue(mctx.supportVision(), "Multimodal context does not report vision support");

            // Programmatically generate a solid red 64x64 pixel bitmap to avoid external dependencies
            int width = 64;
            int height = 64;
            MemorySegment rgbData = arena.allocate(ValueLayout.JAVA_BYTE, width * height * 3);
            for (int i = 0; i < width * height; i++) {
                rgbData.setAtIndex(ValueLayout.JAVA_BYTE, i * 3, (byte) 255);     // Red
                rgbData.setAtIndex(ValueLayout.JAVA_BYTE, i * 3 + 1, (byte) 0);   // Green
                rgbData.setAtIndex(ValueLayout.JAVA_BYTE, i * 3 + 2, (byte) 0);   // Blue
            }

            try (ArgusBitmap bitmap = ArgusBitmap.fromRgb(width, height, rgbData)) {
                String prompt = "<|im_start|>user\n<__media__>\nWhat color is this image?<|im_end|>\n<|im_start|>assistant\n";
                try (ArgusInputChunks chunks = mctx.tokenize(arena, prompt, true, List.of(bitmap))) {
                    int newNPast = context.evalMultimodalChunks(mctx, chunks, 0, 0, 1024, true);
                    assertTrue(newNPast > 0);

                    // Autoregressive response decoding
                    StringBuilder response = new StringBuilder();
                    int currentPos = newNPast;
                    int token = context.sampleToken(0, 0.2f, 1.1f);
                    int limit = 20;

                    while (limit-- > 0) {
                        if (token == 151643 || token == 151645 || token == 128009 || token == 128001 || token == 2) {
                            break;
                        }
                        String piece = context.tokenToPiece(token);
                        if (piece != null && !piece.isEmpty()) {
                            response.append(piece);
                        }

                        MemorySegment nextTokenBuf = arena.allocate(ValueLayout.JAVA_INT, 1);
                        nextTokenBuf.setAtIndex(ValueLayout.JAVA_INT, 0, token);
                        int decodeRes = context.decodeBatch(nextTokenBuf, 1, currentPos, 0, true);
                        if (decodeRes != 0) {
                            break;
                        }
                        currentPos++;
                        token = context.sampleToken(0, 0.2f, 1.1f);
                    }

                    String outStr = response.toString().toLowerCase();
                    System.out.println("  - Visual Query response: " + outStr.trim());
                    assertFalse(outStr.trim().isEmpty(), "Generated response was empty");
                }
            }
        }
    }

    @Test
    public void testCell5_VideoToText() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell 5: Video In -> Text Out...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing at: " + VLM_BASE);
        Assumptions.assumeTrue(Files.exists(VLM_PROJ), "Skipping: VLM Projector missing at: " + VLM_PROJ);
        Assumptions.assumeTrue(Files.exists(TEST_VIDEO), "Skipping: Video test file missing at: " + TEST_VIDEO);
        Assumptions.assumeTrue(isCommandAvailable("ffmpeg"), "Skipping: ffmpeg is not installed.");

        try (Arena arena = Arena.ofConfined();
             ArgusModel baseModel = ArgusModel.load(arena, VLM_BASE, 99, true);
             ArgusContext context = ArgusContext.init(arena, baseModel, ArgusContextConfig.createDefault(8192));
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(arena, baseModel, VLM_PROJ, 4, true)) {

            assertTrue(mctx.supportVideo(), "Multimodal context does not report video support");

            // Extract frame bitmaps and timestamps
            try (ArgusVideo video = ArgusVideo.loadFile(arena, mctx, TEST_VIDEO, 1.0f, 1000);
                 ArgusVideoItem item = new ArgusVideoItem()) {

                int frameCount = 0;
                int nPast = 0;

                while (video.readNext(item) && frameCount < 3) {
                    if (item.bitmap() != null) {
                        String prompt = "<__media__>";
                        try (ArgusInputChunks chunks = mctx.tokenize(arena, prompt, false, List.of(item.bitmap()))) {
                            nPast = context.evalMultimodalChunks(mctx, chunks, nPast, 0, 1024, false);
                            assertTrue(nPast > 0);
                        }
                        frameCount++;
                    } else if (item.text() != null) {
                        System.out.println("  - Read video timestamp: " + item.text());
                    }
                }
                
                assertTrue(frameCount > 0, "No video frames were successfully extracted and evaluated");
                System.out.println("  - Successfully read and evaluated " + frameCount + " frames.");
            }
        }
    }

    @Test
    public void testCell7_AudioToText() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell 7: Audio In -> Text Out...");
        Assumptions.assumeTrue(Files.exists(ASR_MODEL), "Skipping: Whisper base model missing at: " + ASR_MODEL);

        // Generate 1 second of a pure 440Hz sine wave at 16kHz to avoid local media dependencies
        int sampleRate = 16000;
        int nSamples = sampleRate; // 1 second
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pcmSeg = arena.allocate(ValueLayout.JAVA_FLOAT, nSamples);
            for (int i = 0; i < nSamples; i++) {
                float sample = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate) * 0.5f;
                pcmSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, sample);
            }

            try (ArgusAudioContext audioCtx = ArgusAudioContext.init(arena, ASR_MODEL.toAbsolutePath().toString(), 4, 99)) {
                String transcript = audioCtx.transcribe(pcmSeg, nSamples, 1024);
                assertNotNull(transcript);
                System.out.println("  - Transcription output: \"" + transcript.trim() + "\"");
            }
        }
    }

    @Test
    public void testCell_JinaEmbeddings() {
        System.out.println("[MultimodalMatrixTest] Verifying Cell: Jina Embeddings v3...");
        Path jinaModelPath = MODELS_DIR.resolve("jina-embeddings-v3-Q4_K_M.gguf");
        Assumptions.assumeTrue(Files.exists(jinaModelPath), "Skipping: Jina Embeddings v3 model missing at: " + jinaModelPath);

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, jinaModelPath, 99, true)) {

            // Create context configuration with embeddings enabled
            ArgusContextConfig config = new ArgusContextConfig(null, 512, 4, 0, 0, 0, false, true);

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                // Jina v3 prompt prefix task
                String text = "text-matching: This is a semantic text embeddings test pattern for libargus Panama FFM bindings.";
                MemorySegment textSeg = arena.allocateFrom(text);

                // Tokenize input prompt
                int maxTokens = 512;
                MemorySegment tokenBuf = arena.allocate(ValueLayout.JAVA_INT, maxTokens);
                int nTokens = context.tokenize(textSeg, tokenBuf, true);
                assertTrue(nTokens > 0, "Tokenized output token count must be greater than 0");

                // Evaluate batch
                int decodeRes = context.decodeBatch(tokenBuf, nTokens, 0, 0, false);
                assertEquals(0, decodeRes, "Batch evaluation for embeddings decoding failed");

                // Expected Jina Embeddings v3 output dimension (1024 floats)
                int expectedDim = 1024;
                MemorySegment embeddingsBuf = arena.allocate(ValueLayout.JAVA_FLOAT, expectedDim);

                // Retrieve embedding vector
                int nFloats = context.getEmbeddings(0, embeddingsBuf, expectedDim);
                assertEquals(expectedDim, nFloats, "Retrieved embedding vector dimension mismatch");

                // Fanged test: passing a buffer size that is too small must return -2 (error)
                MemorySegment tooSmallBuf = arena.allocate(ValueLayout.JAVA_FLOAT, expectedDim - 1);
                int resTooSmall = context.getEmbeddings(0, tooSmallBuf, expectedDim - 1);
                assertEquals(-2, resTooSmall, "Retrieving embeddings with a buffer size too small must return -2 error");

                // Verify the values are not all zeros (validating actual content generation)
                boolean hasNonZero = false;
                for (int i = 0; i < expectedDim; i++) {
                    float val = embeddingsBuf.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    if (val != 0.0f) {
                        hasNonZero = true;
                        break;
                    }
                }
                assertTrue(hasNonZero, "Retrieved embedding vector contains only zeros");
                System.out.println("  - Jina Embeddings test completed. Dim: " + nFloats + ", Has non-zero: " + hasNonZero);
            }
        } catch (Exception e) {
            fail("Jina embeddings test failed with exception", e);
        }
    }

    @Test
    public void testAutoChunkingPrefill() {
        System.out.println("[MultimodalMatrixTest] Verifying auto-chunking of large prefill...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing");

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, VLM_BASE, 99, true)) {
            
            // n_batch will be configured to 128 (matching contextLength of 128)
            ArgusContextConfig config = new ArgusContextConfig.Builder(128)
                .cpuThreads(4)
                .build();

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                // Allocate a prompt batch larger than n_batch (e.g., 200 tokens)
                int nTokens = 200;
                MemorySegment tokensSeg = arena.allocate(ValueLayout.JAVA_INT, nTokens);
                for (int i = 0; i < nTokens; i++) {
                    tokensSeg.setAtIndex(ValueLayout.JAVA_INT, i, 1); // Fill with token ID 1
                }

                // Evaluate prompt batch. This would crash or error out under old logic since 200 > 128
                int res = context.decodeBatch(tokensSeg, nTokens, 0, 0, false);
                assertEquals(0, res, "Decode batch failed with code: " + res);
                System.out.println("  - Auto-chunking test completed. Evaluated " + nTokens + " tokens successfully (n_batch = 128).");
            }
        } catch (Exception e) {
            fail("Auto-chunking test failed with exception", e);
        }
    }

    @Test
    public void testPrefillCancellation() {
        System.out.println("[MultimodalMatrixTest] Verifying early prefill cancellation...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing");

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, VLM_BASE, 99, true)) {
            
            ArgusContextConfig config = new ArgusContextConfig.Builder(256)
                .cpuThreads(4)
                .build();

            try (ArgusContext context = ArgusContext.init(arena, model, config);
                 ArgusAbortFlag abortFlag = new ArgusAbortFlag()) {
                int nTokens = 50;
                MemorySegment tokensSeg = arena.allocate(ValueLayout.JAVA_INT, nTokens);
                for (int i = 0; i < nTokens; i++) {
                    tokensSeg.setAtIndex(ValueLayout.JAVA_INT, i, 1);
                }

                // Signal abort prior to execution
                abortFlag.abort();
                assertTrue(abortFlag.isAborted());

                // Evaluate prompt batch with abortFlag set. Should abort immediately and return -2
                int res = context.decodeBatch(tokensSeg, nTokens, 0, 0, false, abortFlag);
                assertEquals(-2, res, "Expected status -2 (Aborted) but got: " + res);
                System.out.println("  - Early cancellation test completed. Aborted successfully with code: " + res);
            }
        } catch (Exception e) {
            fail("Cancellation test failed with exception", e);
        }
    }

    @Test
    public void testContextCapacityBatchClamping() {
        System.out.println("[MultimodalMatrixTest] Verifying batch size clamping for context slot capacity limit...");
        Assumptions.assumeTrue(Files.exists(VLM_BASE), "Skipping: VLM Base model missing");

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, VLM_BASE, 99, true)) {
            
            // Set contextLength to 1024. With seq_max = 4, the slot capacity is 256 cells.
            // Under the old logic, n_batch would be set to 1024.
            // Under the new logic, n_batch is clamped to 256.
            ArgusContextConfig config = new ArgusContextConfig.Builder(1024)
                .cpuThreads(4)
                .build();

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                // We evaluate a prompt of 256 tokens (exactly matching the slot capacity ceiling).
                // Under old logic, n_batch would be set to 1024, which exceeds the slot capacity of 256.
                // Under new logic, n_batch is clamped to 256 to match the slot capacity, allowing a successful evaluation.
                int nTokens = 256;
                MemorySegment tokensSeg = arena.allocate(ValueLayout.JAVA_INT, nTokens);
                for (int i = 0; i < nTokens; i++) {
                    tokensSeg.setAtIndex(ValueLayout.JAVA_INT, i, 1);
                }

                int res = context.decodeBatch(tokensSeg, nTokens, 0, 0, false);
                assertEquals(0, res, "Decode batch failed with code: " + res);
                System.out.println("  - Slot capacity batch clamping test passed successfully!");
            }
        } catch (Exception e) {
            fail("Slot capacity batch clamping test failed with exception", e);
        }
    }

    private static boolean isCommandAvailable(String cmd) {
        try {
            Process process = new ProcessBuilder(cmd, "-version").start();
            process.destroy();
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
