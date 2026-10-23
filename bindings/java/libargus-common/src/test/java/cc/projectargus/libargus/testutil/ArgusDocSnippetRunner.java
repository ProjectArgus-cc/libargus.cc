package cc.projectargus.libargus.testutil;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * In-memory Markdown Java snippet extractor, compiler, and execution engine.
 *
 * <p>Extracts fenced {@code ```java} blocks from repository Markdown documentation, compiles them
 * in-memory against the active module's test classpath via {@link ToolProvider#getSystemJavaCompiler()},
 * and executes them while capturing {@code System.out} for deterministic verification.
 */
public final class ArgusDocSnippetRunner {

    private static final Pattern CLASS_DECL_PATTERN =
            Pattern.compile("(?:public\\s+)?(?:final\\s+)?class\\s+([A-Za-z0-9_]+)");

    private ArgusDocSnippetRunner() {}

    /**
     * Represents a single fenced {@code ```java} block extracted from a Markdown file.
     *
     * @param markdownFile path to the source Markdown document
     * @param index        0-based index of the Java block within the document
     * @param startLine    1-based starting line number of the code block in the Markdown file
     * @param heading      nearest preceding Markdown heading (or {@code "<root>"})
     * @param code         raw Java source inside the fenced block
     */
    public record JavaSnippet(
            Path markdownFile,
            int index,
            int startLine,
            String heading,
            String code) {

        /**
         * Returns {@code true} if this snippet declares a top-level {@code class}.
         */
        public boolean hasClassDeclaration() {
            return CLASS_DECL_PATTERN.matcher(code).find();
        }

        /**
         * Extracts the declared class name if present, or throws {@link IllegalStateException}.
         */
        public String declaredClassName() {
            Matcher m = CLASS_DECL_PATTERN.matcher(code);
            if (!m.find()) {
                throw new IllegalStateException(
                        "No class declaration found in snippet #" + index + " (" + heading + ") at "
                                + markdownFile + ":" + startLine);
            }
            return m.group(1);
        }
    }

    /**
     * Extracts all fenced {@code ```java} code blocks from the given Markdown file in document order.
     */
    public static List<JavaSnippet> extractJavaSnippets(Path markdownFile) throws IOException {
        List<String> lines = Files.readAllLines(markdownFile, StandardCharsets.UTF_8);
        List<JavaSnippet> snippets = new ArrayList<>();
        String currentHeading = "<root>";
        boolean inJavaBlock = false;
        int blockStartLine = 0;
        StringBuilder blockBuilder = new StringBuilder();

        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            String trimmed = line.trim();
            if (!inJavaBlock) {
                if (trimmed.startsWith("#")) {
                    currentHeading = trimmed.replaceFirst("^#+\\s*", "").trim();
                } else if (trimmed.equals("```java")) {
                    inJavaBlock = true;
                    blockStartLine = i + 2; // 1-based first line inside the fence
                    blockBuilder.setLength(0);
                }
            } else {
                if (trimmed.equals("```")) {
                    inJavaBlock = false;
                    snippets.add(new JavaSnippet(
                            markdownFile,
                            snippets.size(),
                            blockStartLine,
                            currentHeading,
                            blockBuilder.toString()));
                } else {
                    blockBuilder.append(line).append('\n');
                }
            }
        }

        if (inJavaBlock) {
            throw new IllegalStateException("Unclosed ```java block in " + markdownFile + " starting at line " + blockStartLine);
        }
        return Collections.unmodifiableList(snippets);
    }

    /**
     * Wraps a statement fragment (a snippet without a {@code class} declaration) into a complete
     * compilable Java class with {@code public static void main(String[] args) throws Exception}.
     * Any leading {@code import ...;} statements inside {@code fragmentCode} are automatically hoisted.
     */
    public static String wrapFragmentAsMainClass(
            String className,
            String fragmentCode,
            String extraPrelude,
            String extraPostlude,
            String extraClassMembers) {
        List<String> hoistedImports = new ArrayList<>();
        StringBuilder bodyLines = new StringBuilder(fragmentCode.length());
        for (String line : fragmentCode.split("\n", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith("import ") && trimmed.endsWith(";")) {
                hoistedImports.add(trimmed);
            } else {
                bodyLines.append(line).append('\n');
            }
        }

        StringBuilder sb = new StringBuilder(fragmentCode.length() + 1024);
        sb.append("import cc.projectargus.libargus.*;\n");
        sb.append("import java.nio.*;\n");
        sb.append("import java.nio.charset.*;\n");
        sb.append("import java.nio.file.*;\n");
        sb.append("import java.util.*;\n");
        sb.append("import java.util.concurrent.*;\n");
        for (String imp : hoistedImports) {
            sb.append(imp).append('\n');
        }
        sb.append("public class ").append(className).append(" {\n");
        sb.append("    static float[] loadNormalized16kHzAudio() {\n");
        sb.append("        float[] pcm = new float[16000];\n");
        sb.append("        for (int i = 0; i < pcm.length; i++) {\n");
        sb.append("            pcm[i] = (float) (Math.sin(2.0 * Math.PI * 440.0 * i / 16000.0) * 0.25);\n");
        sb.append("        }\n");
        sb.append("        return pcm;\n");
        sb.append("    }\n");
        if (extraClassMembers != null && !extraClassMembers.isBlank()) {
            sb.append(extraClassMembers).append('\n');
        }
        sb.append("    public static void main(String[] args) throws Exception {\n");
        sb.append("        ArgusBackend.init();\n");
        if (extraPrelude != null && !extraPrelude.isBlank()) {
            sb.append(extraPrelude).append('\n');
        }
        sb.append(bodyLines).append('\n');
        if (extraPostlude != null && !extraPostlude.isBlank()) {
            sb.append(extraPostlude).append('\n');
        }
        sb.append("    }\n");
        sb.append("}\n");
        return sb.toString();
    }

    public static String wrapFragmentAsMainClass(String className, String fragmentCode, String extraPrelude) {
        return wrapFragmentAsMainClass(className, fragmentCode, extraPrelude, null, null);
    }

    /**
     * Compiles a complete Java class in memory against the active test classpath and returns its {@link Class}.
     */
    public static Class<?> compileClass(String originDescription, String className, String sourceCode) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException("No system JavaCompiler available (ensure tests run on a full JDK)");
        }

        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        StandardJavaFileManager stdFileManager =
                compiler.getStandardFileManager(diagnostics, Locale.ROOT, StandardCharsets.UTF_8);

        InMemoryClassFileManager memFileManager = new InMemoryClassFileManager(stdFileManager);
        JavaFileObject sourceObject = new InMemorySourceFileObject(className, sourceCode);

        List<String> options = List.of(
                "-classpath", System.getProperty("java.class.path"),
                "-proc:none");

        Boolean success = compiler.getTask(
                null,
                memFileManager,
                diagnostics,
                options,
                null,
                List.of(sourceObject)).call();

        if (!Boolean.TRUE.equals(success)) {
            StringBuilder err = new StringBuilder();
            err.append("Failed to compile documentation snippet [").append(originDescription).append("]:\n");
            for (Diagnostic<? extends JavaFileObject> d : diagnostics.getDiagnostics()) {
                err.append("  Line ").append(d.getLineNumber())
                        .append(": ").append(d.getMessage(Locale.ROOT)).append('\n');
            }
            err.append("--- Source (").append(className).append(") ---\n").append(sourceCode);
            throw new AssertionError(err.toString());
        }

        try {
            return memFileManager.getClassLoader(null).loadClass(className);
        } catch (ClassNotFoundException e) {
            throw new AssertionError("Compiled class not found: " + className, e);
        }
    }

    /**
     * Compiles a Java snippet (either a full class or a wrapped fragment), invokes its
     * {@code public static void main(String[] args)}, and returns the captured {@code System.out} text.
     */
    public static String compileAndRunMain(
            JavaSnippet snippet,
            UnaryOperator<String> sourceTransformer,
            String... args) throws Exception {
        return compileAndRunFragment(snippet, sourceTransformer, null, null, null, args);
    }

    /**
     * Compiles a Java snippet with optional prelude, postlude, and helper class members, invokes
     * {@code public static void main(String[] args)}, and returns captured {@code System.out} text.
     */
    public static String compileAndRunFragment(
            JavaSnippet snippet,
            UnaryOperator<String> sourceTransformer,
            String extraPrelude,
            String extraPostlude,
            String extraClassMembers,
            String... args) throws Exception {
        String transformed = sourceTransformer != null ? sourceTransformer.apply(snippet.code()) : snippet.code();
        String className;
        String fullSource;
        Matcher m = CLASS_DECL_PATTERN.matcher(transformed);
        if (m.find() && extraPrelude == null && extraPostlude == null && extraClassMembers == null) {
            className = m.group(1);
            fullSource = transformed;
        } else {
            className = "DocSnippet_" + snippet.index();
            fullSource = wrapFragmentAsMainClass(
                    className, transformed, extraPrelude, extraPostlude, extraClassMembers);
        }

        String origin = snippet.markdownFile().getFileName() + ":" + snippet.startLine()
                + " (" + snippet.heading() + ")";
        Class<?> compiledClass = compileClass(origin, className, fullSource);
        return invokeMainCapturingStdout(origin, compiledClass, args);
    }

    /**
     * Escapes a {@link Path} for safe interpolation inside a Java string literal.
     */
    public static String escapeJavaPath(Path path) {
        return path.toAbsolutePath().normalize().toString().replace("\\", "\\\\");
    }

    /**
     * Compiles and executes all 4 core Java quickstart snippets (indices 0..3) in {@code README.md}
     * against the active binding module on the test classpath.
     */
    public static void verifyReadmeSnippets(Path tempDir) throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path readme = root.resolve("README.md");
        List<JavaSnippet> snippets = extractJavaSnippets(readme);
        if (snippets.size() != 5) {
            throw new AssertionError("Expected 5 Java snippets in README.md, found " + snippets.size());
        }

        String tinyGguf = escapeJavaPath(ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir));
        String tinyEmbedGguf = escapeJavaPath(root.resolve("tests/data/tiny-embed.gguf"));
        String tinyMmprojGguf = escapeJavaPath(root.resolve("tests/data/tiny-mmproj.gguf"));
        String catPng = escapeJavaPath(ArgusSyntheticTestFixtures.generateTestPng(tempDir, "cat.png"));
        String tinyWhisperBin = escapeJavaPath(ArgusSyntheticTestFixtures.generateTinyWhisperModel(tempDir));

        // 0. QuickstartText (1. Streaming LLM Text Generation)
        compileAndRunMain(snippets.get(0), src -> src
                .replace("models/llama-3-8b.gguf", tinyGguf)
                .replace("Builder(4096)", "Builder(64)")
                .replace("KV_TYPE_Q4_0", "KV_TYPE_F16")
                .replace(", 256,", ", 4,")
                .replace(", 128,", ", 4,"));

        // 1. Semantic Text Embeddings
        String embedOut = compileAndRunMain(snippets.get(1), src -> src
                .replace("models/jina-embeddings-v3.gguf", tinyEmbedGguf)
                .replace("Builder(512)", "Builder(64)"));
        if (!embedOut.contains("Extracted 32-d vector.")) {
            throw new AssertionError("Unexpected README embedding snippet output: " + embedOut);
        }

        // 2. Multimodal Vision & Audio Prompting
        String visionOut = compileAndRunMain(snippets.get(2), src -> src
                .replace("models/qwen2-vl-7b-it.gguf", tinyGguf)
                .replace("Builder(8192)", "Builder(64)")
                .replace("1024, true)", "64, true)")
                .replace("models/qwen2-vl-7b-it.mmproj", tinyMmprojGguf)
                .replace("media/cat.png", catPng));
        if (!visionOut.contains("First generated piece:")) {
            throw new AssertionError("Unexpected README multimodal snippet output: " + visionOut);
        }

        // 3. Whisper Speech-to-Text (ASR)
        String asrOut = compileAndRunMain(snippets.get(3), src -> src
                .replace("models/whisper-base.en.gguf", tinyWhisperBin));
        if (!asrOut.contains("argus")) {
            throw new AssertionError("Unexpected README Whisper ASR snippet output: '" + asrOut + "'");
        }
    }

    /**
     * Compiles and executes all 13 core Java snippets (indices 0..12) in {@code docs/quickstart-java.md}
     * against the active binding module on the test classpath.
     */
    public static void verifyQuickstartJavaSnippets(Path tempDir) throws Exception {
        Path root = ArgusSyntheticTestFixtures.resolveRepoRoot();
        Path quickstartJava = root.resolve("docs/quickstart-java.md");
        List<JavaSnippet> snippets = extractJavaSnippets(quickstartJava);
        if (snippets.size() != 14) {
            throw new AssertionError("Expected 14 Java snippets in docs/quickstart-java.md, found " + snippets.size());
        }

        String tinyGguf = escapeJavaPath(ArgusSyntheticTestFixtures.generateTinyCausalModel(tempDir));
        String tinyEmbedGguf = escapeJavaPath(root.resolve("tests/data/tiny-embed.gguf"));
        String tinyMmprojGguf = escapeJavaPath(root.resolve("tests/data/tiny-mmproj.gguf"));
        String tinyVideoY4m = escapeJavaPath(root.resolve("tests/data/tiny-video.y4m"));
        String diagramPng = escapeJavaPath(ArgusSyntheticTestFixtures.generateTestPng(tempDir, "diagram.png"));
        String tinyWhisperBin = escapeJavaPath(ArgusSyntheticTestFixtures.generateTinyWhisperModel(tempDir));
        String tinyVocoderGguf = escapeJavaPath(ArgusSyntheticTestFixtures.generateTinyVocoderModel(tempDir));

        // 0. 2.1 High-Level Streaming & Early-Stopping Predicates (TextGenerationExample)
        compileAndRunMain(snippets.get(0), src -> src
                .replace("models/llama-3-8b.gguf", tinyGguf)
                .replace("Builder(4096)", "Builder(64)")
                .replace("KV_TYPE_Q4_0", "KV_TYPE_F16")
                .replace("256,", "4,")
                .replace("512,", "4,"));

        // 1. 2.2 Cross-Thread Cooperative Cancellation (ArgusAbortFlag)
        compileAndRunFragment(
                snippets.get(1),
                src -> src.replace("1024,", "4,"),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false)) {\n"
                        + "ArgusContextConfig config = new ArgusContextConfig.Builder(64).build();\n"
                        + "String largePrompt = \"Hello Argus\";\n"
                        + "ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder().temperature(0.0f).build();",
                "}",
                null);

        // 2. 3. Low-Level Autoregressive Loop, Persistent Samplers & Logit Bias
        compileAndRunFragment(
                snippets.get(2),
                src -> src.replace("step < 128", "step < 4"),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                        + "     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(64).build())) {",
                "}",
                null);

        // 3. 4. Semantic Text Embeddings (EmbeddingsExample)
        String embedOut = compileAndRunMain(snippets.get(3), src -> src
                .replace("models/jina-embeddings-v3-Q4_K_M.gguf", tinyEmbedGguf)
                .replace("Builder(512)", "Builder(64)"));
        if (!embedOut.contains("Embedding dimension: 32")) {
            throw new AssertionError("Unexpected EmbeddingsExample output: " + embedOut);
        }

        // 4. 5.1 Vision & Audio Prompting (VisionPromptExample)
        compileAndRunMain(snippets.get(4), src -> src
                .replace("models/qwen2-vl-7b-it.gguf", tinyGguf)
                .replace("Builder(8192)", "Builder(64)")
                .replace("1024, true)", "64, true)")
                .replace("models/qwen2-vl-7b-it.mmproj", tinyMmprojGguf)
                .replace("media/diagram.png", diagramPng)
                .replace("step < 256", "step < 4"));

        // 5. 5.2 Frame-by-Frame Video Stream Processing (ArgusVideo & ArgusVideoItem)
        boolean hasFfmpeg = isCommandAvailable("ffprobe") && isCommandAvailable("ffmpeg");
        try {
            String videoOut = compileAndRunFragment(
                    snippets.get(5),
                    src -> src.replace("media/clip.mp4", tinyVideoY4m),
                    "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                            + "     ArgusMultimodalContext mctx = ArgusMultimodalContext.init(model, Path.of(\"" + tinyMmprojGguf + "\"), 2, false)) {",
                    "}",
                    null);
            if (!videoOut.contains("Timestamp marker:")) {
                throw new AssertionError("Expected timestamp markers from ArgusVideo snippet, got: " + videoOut);
            }
        } catch (AssertionError err) {
            if (hasFfmpeg || !(err.getCause() instanceof cc.projectargus.libargus.ArgusNativeException)) {
                throw err;
            }
        }

        // 6. 6.1 Whisper Audio Transcription (ArgusAudioContext)
        String whisperOut = compileAndRunMain(snippets.get(6), src -> src
                .replace("models/whisper-base.en.gguf", tinyWhisperBin));
        if (!whisperOut.contains("Transcript:") || !whisperOut.contains("argus")) {
            throw new AssertionError("Unexpected Whisper snippet output: " + whisperOut);
        }

        // 7. 6.2 Speech-LLM Text-to-Speech (synthesizeSpeech)
        String ttsOut = compileAndRunMain(snippets.get(7), src -> src
                .replace("models/outetts-0.2-500m.gguf", tinyGguf)
                .replace("models/wavtokenizer-large-75token.gguf", tinyVocoderGguf)
                .replace("Builder(4096)", "Builder(256)"));
        if (!ttsOut.contains("Synthesized ") || !ttsOut.contains(" audio samples at 24kHz.")) {
            throw new AssertionError("Unexpected TTS snippet output: " + ttsOut);
        }

        // 8. 7.1 Zero-Rollback Sequence Slot Forking (forkSlot / freeSlot)
        compileAndRunFragment(
                snippets.get(8),
                src -> src.replace("Builder(4096)", "Builder(64)"),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false)) {\n"
                        + "ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder().temperature(0.0f).build();",
                "}",
                null);

        // 9. 7.2 Sliding-Window Ring-Buffer Eviction (clearCacheSlot & shiftCacheSlot)
        compileAndRunFragment(
                snippets.get(9),
                src -> src
                        .replace("int nKeep = 64;", "int nKeep = 2;")
                        .replace("int nDiscard = 256;", "int nDiscard = 2;"),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                        + "     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(64).build())) {\n"
                        + "int[] prefill = new int[]{1, 2, 3, 4, 5, 6};\n"
                        + "context.decodeBatch(prefill, prefill.length, 0, 0, false);",
                "if (context.getSeqPosMax(0) != 3) throw new AssertionError(\"Expected shifted max pos 3, got \" + context.getSeqPosMax(0));\n}",
                null);

        // 10. 8.1 Pre-Allocation VRAM Budgeting & GGUF Metadata
        compileAndRunFragment(
                snippets.get(10),
                UnaryOperator.identity(),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false)) {",
                "if (weightBytes <= 0 || kvPerToken <= 0 || totalVramFor32k <= weightBytes || !\"llama\".equals(arch) || allMeta.isEmpty()) {\n"
                        + "    throw new AssertionError(\"Invalid VRAM/metadata values\");\n"
                        + "}\n}",
                null);

        // 11. 8.2 Native Logging Control & Custom Log Routing
        compileAndRunFragment(
                snippets.get(11),
                UnaryOperator.identity(),
                null,
                "ArgusBackend.setLogLevel(ArgusLogLevel.WARN);\nArgusBackend.setLogCallback(null);",
                null);

        // 12. 8.3 Hardware-Synchronized Performance Telemetry (ArgusPerfTimings)
        String perfOut = compileAndRunFragment(
                snippets.get(12),
                UnaryOperator.identity(),
                "try (ArgusModel model = ArgusModel.load(Path.of(\"" + tinyGguf + "\"), 0, false);\n"
                        + "     ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(64).build())) {\n"
                        + "context.decodeBatch(new int[]{1, 2}, 2, 0, 0, true);",
                "}",
                null);
        if (!perfOut.contains("Prefill:") || !perfOut.contains("Decode:")) {
            throw new AssertionError("Unexpected PerfTimings snippet output: " + perfOut);
        }
    }

    private static boolean isCommandAvailable(String command) {
        try {
            Process p = new ProcessBuilder(command, "-version")
                    .redirectErrorStream(true)
                    .start();
            p.getInputStream().readAllBytes();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Invokes {@code public static void main(String[])} on {@code compiledClass} while capturing
     * {@code System.out} into a UTF-8 string.
     */
    public static String invokeMainCapturingStdout(String originDescription, Class<?> compiledClass, String... args)
            throws Exception {
        Method mainMethod = compiledClass.getMethod("main", String[].class);
        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try (PrintStream intercept = new PrintStream(captured, true, StandardCharsets.UTF_8)) {
            System.setOut(intercept);
            try {
                mainMethod.invoke(null, (Object) (args != null ? args : new String[0]));
            } catch (InvocationTargetException e) {
                Throwable cause = e.getCause() != null ? e.getCause() : e;
                if (cause instanceof Exception ex) {
                    throw new AssertionError("Execution failed in snippet [" + originDescription + "]: " + ex, ex);
                }
                if (cause instanceof Error err) {
                    throw err;
                }
                throw new AssertionError("Execution failed in snippet [" + originDescription + "]", cause);
            }
        } finally {
            System.setOut(originalOut);
        }
        return captured.toString(StandardCharsets.UTF_8);
    }

    private static final class InMemorySourceFileObject extends SimpleJavaFileObject {
        private final String sourceCode;

        InMemorySourceFileObject(String className, String sourceCode) {
            super(URI.create("string:///" + className.replace('.', '/') + Kind.SOURCE.extension), Kind.SOURCE);
            this.sourceCode = sourceCode;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return sourceCode;
        }
    }

    private static final class InMemoryBytecodeFileObject extends SimpleJavaFileObject {
        private final ByteArrayOutputStream bos = new ByteArrayOutputStream();

        InMemoryBytecodeFileObject(String className, Kind kind) {
            super(URI.create("mem:///" + className.replace('.', '/') + kind.extension), kind);
        }

        @Override
        public OutputStream openOutputStream() {
            return bos;
        }

        byte[] getBytes() {
            return bos.toByteArray();
        }
    }

    private static final class InMemoryClassFileManager extends ForwardingJavaFileManager<StandardJavaFileManager> {
        private final Map<String, InMemoryBytecodeFileObject> classBytes = new HashMap<>();

        InMemoryClassFileManager(StandardJavaFileManager fileManager) {
            super(fileManager);
        }

        @Override
        public JavaFileObject getJavaFileForOutput(
                Location location, String className, JavaFileObject.Kind kind, FileObject sibling) {
            InMemoryBytecodeFileObject file = new InMemoryBytecodeFileObject(className, kind);
            classBytes.put(className, file);
            return file;
        }

        @Override
        public ClassLoader getClassLoader(Location location) {
            return new ClassLoader(ArgusDocSnippetRunner.class.getClassLoader()) {
                @Override
                protected Class<?> findClass(String name) throws ClassNotFoundException {
                    InMemoryBytecodeFileObject file = classBytes.get(name);
                    if (file != null) {
                        byte[] bytes = file.getBytes();
                        return defineClass(name, bytes, 0, bytes.length);
                    }
                    return super.findClass(name);
                }
            };
        }
    }
}
