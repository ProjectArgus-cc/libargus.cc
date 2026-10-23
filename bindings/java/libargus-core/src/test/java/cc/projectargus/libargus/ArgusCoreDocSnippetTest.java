package cc.projectargus.libargus;

import static org.junit.jupiter.api.Assertions.*;

import cc.projectargus.libargus.testutil.ArgusDocSnippetRunner;
import cc.projectargus.libargus.testutil.ArgusSyntheticTestFixtures;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Executable Markdown documentation doctest suite and synthetic fixture verification for
 * {@code :libargus-core} (Project Panama FFM).
 */
class ArgusCoreDocSnippetTest {

    @TempDir
    Path tempDir;

    @Test
    void testReadmeSnippetsCompileAndExecuteOnPanamaFfm() throws Exception {
        ArgusDocSnippetRunner.verifyReadmeSnippets(tempDir);
    }

    @Test
    void testQuickstartJavaSnippetsCompileAndExecuteOnPanamaFfm() throws Exception {
        ArgusDocSnippetRunner.verifyQuickstartJavaSnippets(tempDir);
    }

    @Test
    void testSyntheticFixturesFullSubsystemCoverageOnPanamaFfm() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path tinyModelPath = ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir);
        Path omniProjPath = ArgusSyntheticTestFixtures.generateTinyOmniProjector(tempDir);
        Path whisperPath = ArgusSyntheticTestFixtures.generateTinyWhisperModel(tempDir);
        Path vocoderPath = ArgusSyntheticTestFixtures.generateTinyVocoderModel(tempDir);
        Path pngPath = ArgusSyntheticTestFixtures.generateTestPng(tempDir, "omni.png");
        Path wavPath = ArgusSyntheticTestFixtures.generateTestWav16kHz(tempDir, "omni.wav");

        ArgusBackend.init();

        // 1. Unified Omni Projector (Vision + Audio in a single prompt)
        try (ArgusModel model = ArgusModel.load(tinyModelPath, 0, false);
             ArgusContext ctx = ArgusContext.init(model, new ArgusContextConfig.Builder(128).build());
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(model, omniProjPath, 2, false);
             ArgusBitmap image = ArgusBitmap.loadFile(mctx, pngPath, false);
             ArgusBitmap audio = ArgusBitmap.loadFile(mctx, wavPath, false)) {

            assertTrue(mctx.supportVision(), "Omni projector must report vision support");
            assertTrue(mctx.supportAudio(), "Omni projector must report audio support");
            assertEquals(16000, mctx.getAudioSampleRate());

            try (ArgusInputChunks chunks = mctx.tokenize(
                    "Image: <__media__> Audio: <__media__>", true, List.of(image, audio))) {
                int nPast = ctx.evalMultimodalChunks(mctx, chunks, 0, 0, 64, true);
                assertTrue(nPast > 0, "Expected positive KV cursor after multimodal vision+audio evaluation");
                int tok = ctx.sampleToken(0, new ArgusSamplerConfig.Builder().temperature(0.0f).build());
                assertTrue(tok >= 0);
            }
        }

        // 2. Whisper ASR synthetic model
        try (ArgusAudioContext whisper = ArgusAudioContext.init(whisperPath, 2, 0)) {
            float[] pcm = new float[16000];
            String text = whisper.transcribe(pcm, pcm.length, 256);
            assertNotNull(text);
            assertTrue(text.contains("argus"), "Expected synthetic Whisper model to transcribe 'argus', got: " + text);
        }

        // 3. Synthetic 1,282-dim Vocoder + Speech Synthesis
        try (ArgusModel ttsModel = ArgusModel.load(tinyModelPath, 0, false);
             ArgusModel vocoder = ArgusModel.load(vocoderPath, 0, false);
             ArgusContext ttsCtx = ArgusContext.init(ttsModel, new ArgusContextConfig.Builder(256).build())) {
            float[] samples = ttsCtx.synthesizeSpeech(vocoder, "Synthetic vocoder test", 42, 24000);
            assertNotNull(samples);
            assertTrue(samples.length > 0, "Expected synthesized waveform samples > 0");
            boolean hasNonZero = false;
            for (float s : samples) {
                assertTrue(Float.isFinite(s), "Synthesized PCM sample must be finite");
                if (Math.abs(s) > 1e-7f) {
                    hasNonZero = true;
                }
            }
            assertTrue(hasNonZero, "Expected non-zero acoustic energy from overlap-add vocoder");
        }
    }
}
