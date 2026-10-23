package cc.projectargus.libargus;

import static org.junit.jupiter.api.Assertions.*;

import cc.projectargus.libargus.testutil.ArgusDocSnippetRunner;
import cc.projectargus.libargus.testutil.ArgusDocSnippetRunner.JavaSnippet;
import cc.projectargus.libargus.testutil.ArgusSyntheticTestFixtures;
import java.nio.file.Path;
import java.util.List;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Executable Markdown documentation doctest suite and synthetic fixture verification for
 * {@code :libargus-jni} (Standard JNI / JDK 17+ / Android).
 */
class ArgusJniDocSnippetTest {

    @TempDir
    Path tempDir;

    @Test
    void testReadmeSnippetsCompileAndExecuteOnJni() throws Exception {
        ArgusDocSnippetRunner.verifyReadmeSnippets(tempDir);
    }

    @Test
    void testQuickstartJavaSnippetsCompileAndExecuteOnJni() throws Exception {
        ArgusDocSnippetRunner.verifyQuickstartJavaSnippets(tempDir);
    }

    @Test
    void testQuickstartAndroidJniSnippetsCompileAndExecuteOnJni() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path androidMd = root.resolve("docs/quickstart-android-jni.md");
        List<JavaSnippet> snippets = ArgusDocSnippetRunner.extractJavaSnippets(androidMd);
        assertEquals(3, snippets.size(), "Expected 3 Java snippets in docs/quickstart-android-jni.md");

        String tinyGguf = ArgusDocSnippetRunner.escapeJavaPath(
                ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir));
        String tinyWhisperBin = ArgusDocSnippetRunner.escapeJavaPath(
                ArgusSyntheticTestFixtures.generateTinyWhisperModel(tempDir));

        // 0. Section 2: Safe File Descriptor Model Loading on Android (ArgusModel.loadFd)
        // Strip android.* imports and supply lightweight POSIX-backed ParcelFileDescriptor / ContentResolver / Uri shims
        JavaSnippet fdSnippet = snippets.get(0);
        String fdMethodBody = fdSnippet.code()
                .replace("import android.content.ContentResolver;\n", "")
                .replace("import android.net.Uri;\n", "")
                .replace("import android.os.ParcelFileDescriptor;\n", "")
                .replace("import cc.projectargus.libargus.*;\n", "")
                .replace("Builder(4096)", "Builder(64)")
                .replace(", 128,", ", 4,");

        String androidShims = """
                static final class Uri {
                    final String path;
                    Uri(String path) { this.path = path; }
                }
                static final class ParcelFileDescriptor implements AutoCloseable {
                    private final java.io.RandomAccessFile raf;
                    private final int fd;
                    ParcelFileDescriptor(String path) throws Exception {
                        this.raf = new java.io.RandomAccessFile(path, "r");
                        java.lang.reflect.Field f = java.io.FileDescriptor.class.getDeclaredField("fd");
                        f.setAccessible(true);
                        this.fd = f.getInt(this.raf.getFD());
                    }
                    int getFd() { return fd; }
                    @Override public void close() throws java.io.IOException { raf.close(); }
                }
                static final class ContentResolver {
                    ParcelFileDescriptor openFileDescriptor(Uri uri, String mode) throws Exception {
                        return new ParcelFileDescriptor(uri.path);
                    }
                }
                """ + fdMethodBody;

        ArgusDocSnippetRunner.compileAndRunFragment(
                new JavaSnippet(fdSnippet.markdownFile(), 0, fdSnippet.startLine(), fdSnippet.heading(),
                        "new DocSnippet_0().runInferenceFromContentUri(new ContentResolver(), new Uri(\"" + tinyGguf + "\"));"),
                UnaryOperator.identity(),
                null,
                null,
                androidShims);

        // 1. Section 3: Zero-Copy Off-Heap JNI Overloads (DirectByteBuffer)
        ArgusDocSnippetRunner.compileAndRunFragment(
                snippets.get(1),
                UnaryOperator.identity(),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                        + "     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(64).build())) {",
                "int tok = context.sampleToken(0, new ArgusSamplerConfig.Builder().temperature(0.0f).build());\n"
                        + "if (tok < 0) throw new AssertionError(\"Expected valid token after decodeBatchDirect\");\n}",
                null);

        // 2. Section 4: Mobile Thermal & Power Governor Integration (setNThreads)
        ArgusDocSnippetRunner.compileAndRunFragment(
                snippets.get(2),
                UnaryOperator.identity(),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                        + "     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(64).build());\n"
                        + "     ArgusAudioContext audioContext = ArgusAudioContext.init(Path.of(\"" + tinyWhisperBin + "\"), 2, 0)) {",
                "if (context.getNThreads() != 2 || context.getNThreadsBatch() != 2) {\n"
                        + "    throw new AssertionError(\"Expected 2 threads after thermal scale-down\");\n"
                        + "}\n}",
                null);
    }

    @Test
    void testSyntheticFixturesFullSubsystemCoverageOnJni() throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path tinyModelPath = ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir);
        Path omniProjPath = ArgusSyntheticTestFixtures.generateTinyOmniProjector(tempDir);
        Path whisperPath = ArgusSyntheticTestFixtures.generateTinyWhisperModel(tempDir);
        Path vocoderPath = ArgusSyntheticTestFixtures.generateTinyVocoderModel(tempDir);
        Path pngPath = ArgusSyntheticTestFixtures.generateTestPng(tempDir, "omni.png");
        Path wavPath = ArgusSyntheticTestFixtures.generateTestWav16kHz(tempDir, "omni.wav");

        ArgusBackend.init();

        try (ArgusModel model = ArgusModel.load(tinyModelPath, 0, false);
             ArgusContext ctx = ArgusContext.init(model, new ArgusContextConfig.Builder(128).build());
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(model, omniProjPath, 2, false);
             ArgusBitmap image = ArgusBitmap.loadFile(mctx, pngPath, false);
             ArgusBitmap audio = ArgusBitmap.loadFile(mctx, wavPath, false);
             ArgusInputChunks chunks = mctx.tokenize(
                     "Image: <__media__> Audio: <__media__>", true, List.of(image, audio))) {

            assertTrue(mctx.supportVision());
            assertTrue(mctx.supportAudio());
            assertEquals(16000, mctx.getAudioSampleRate());
            int nPast = ctx.evalMultimodalChunks(mctx, chunks, 0, 0, 64, true);
            assertTrue(nPast > 0);
        }

        try (ArgusAudioContext whisper = ArgusAudioContext.init(whisperPath, 2, 0)) {
            float[] pcm = new float[16000];
            String text = whisper.transcribe(pcm, pcm.length, 256);
            assertTrue(text.contains("argus"));
        }

        try (ArgusModel ttsModel = ArgusModel.load(tinyModelPath, 0, false);
             ArgusModel vocoder = ArgusModel.load(vocoderPath, 0, false);
             ArgusContext ttsCtx = ArgusContext.init(ttsModel, new ArgusContextConfig.Builder(256).build())) {
            float[] samples = ttsCtx.synthesizeSpeech(vocoder, "JNI synthetic vocoder test", 42, 24000);
            assertTrue(samples.length > 0);
        }
    }
}
