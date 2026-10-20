package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Differential verification suite executing identical workloads against both {@code :libargus-core}
 * (Panama FFM) and {@code :libargus-jni} (JNI) inside the same JVM process backed by the unified
 * {@code libargus} shared library.
 */
public class ArgusDifferentialEquivalenceTest {

    private static final Set<String> SHARED_COMMON_CLASSES = Set.of(
        "cc.projectargus.libargus.ArgusLogLevel",
        "cc.projectargus.libargus.ArgusLogCallback",
        "cc.projectargus.libargus.ArgusPerfTimings",
        "cc.projectargus.libargus.ArgusSamplerConfig",
        "cc.projectargus.libargus.ArgusNativeException",
        "cc.projectargus.libargus.ArgusNativeException$ErrorResolver",
        "cc.projectargus.libargus.internal.ArgusNativeLoader",
        "cc.projectargus.libargus.internal.AbstractNativeResource",
        "cc.projectargus.libargus.internal.AbstractNativeResource$HandleAction",
        "cc.projectargus.libargus.internal.ArgusUtf8StreamDecoder",
        "cc.projectargus.libargus.internal.ArgusCommonValidation",
        "cc.projectargus.libargus.internal.ArgusBackendSupport"
    );

    private static URLClassLoader createIsolatedCoreClassLoader() throws Exception {
        String coreJarPath = System.getProperty("argus.core.jar");
        assertNotNull(coreJarPath, "System property argus.core.jar must be set by Gradle test task");
        File jarFile = new File(coreJarPath);
        assertTrue(jarFile.isFile(), "libargus-core jar does not exist at: " + coreJarPath);

        URL[] urls = new URL[] { jarFile.toURI().toURL() };
        ClassLoader parent = ArgusDifferentialEquivalenceTest.class.getClassLoader();

        return new URLClassLoader(urls, parent) {
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
        };
    }

    private static Path findTestModelPath() {
        Path root = Paths.get(System.getProperty("argus.root", "")).toAbsolutePath();
        while (root != null && !Files.exists(root.resolve("tests/data/tiny-embed.gguf"))) {
            root = root.getParent();
        }
        assertNotNull(root, "Could not locate project root containing tests/data/tiny-embed.gguf");
        return root.resolve("tests/data/tiny-embed.gguf");
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testFfmAndJniBitForBitEquivalence() throws Exception {
        ArgusBackend.init();
        Path modelPath = findTestModelPath();

        try (URLClassLoader coreLoader = createIsolatedCoreClassLoader()) {
            Class<?> ffmBackendCls = coreLoader.loadClass("cc.projectargus.libargus.ArgusBackend");
            Class<?> ffmModelCls = coreLoader.loadClass("cc.projectargus.libargus.ArgusModel");
            Class<?> ffmConfigCls = coreLoader.loadClass("cc.projectargus.libargus.ArgusContextConfig");
            Class<?> ffmConfigBuilderCls = coreLoader.loadClass("cc.projectargus.libargus.ArgusContextConfig$Builder");
            Class<?> ffmContextCls = coreLoader.loadClass("cc.projectargus.libargus.ArgusContext");

            // 1. Backend parity
            assertEquals(ArgusBackend.getVersion(), ffmBackendCls.getMethod("getVersion").invoke(null));
            assertEquals(ArgusBackend.getBuildFeatures(), ffmBackendCls.getMethod("getBuildFeatures").invoke(null));
            assertEquals(ArgusBackend.getAvailableBackends(), (List<String>) ffmBackendCls.getMethod("getAvailableBackends").invoke(null));

            // 2. Model introspection, metadata & tokenization parity
            Method ffmModelLoad = ffmModelCls.getMethod("load", Path.class, int.class, boolean.class);
            try (ArgusModel jniModel = ArgusModel.load(modelPath, 0, false);
                 AutoCloseable ffmModel = (AutoCloseable) ffmModelLoad.invoke(null, modelPath, 0, false)) {

                assertEquals((int) ffmModelCls.getMethod("vocabBos").invoke(ffmModel), jniModel.vocabBos());
                assertEquals((int) ffmModelCls.getMethod("vocabEos").invoke(ffmModel), jniModel.vocabEos());
                assertEquals((int) ffmModelCls.getMethod("vocabEot").invoke(ffmModel), jniModel.vocabEot());
                assertEquals((int) ffmModelCls.getMethod("vocabPad").invoke(ffmModel), jniModel.vocabPad());
                assertEquals((int) ffmModelCls.getMethod("vocabNTokens").invoke(ffmModel), jniModel.vocabNTokens());
                assertEquals((int) ffmModelCls.getMethod("nEmbd").invoke(ffmModel), jniModel.nEmbd());
                assertEquals((int) ffmModelCls.getMethod("nEmbdOut").invoke(ffmModel), jniModel.nEmbdOut());
                assertEquals((int) ffmModelCls.getMethod("nEmbdInp").invoke(ffmModel), jniModel.nEmbdInp());
                assertEquals((int) ffmModelCls.getMethod("nCtxTrain").invoke(ffmModel), jniModel.nCtxTrain());
                assertEquals((int) ffmModelCls.getMethod("nLayer").invoke(ffmModel), jniModel.nLayer());
                assertEquals((int) ffmModelCls.getMethod("nHead").invoke(ffmModel), jniModel.nHead());
                assertEquals((int) ffmModelCls.getMethod("nHeadKv").invoke(ffmModel), jniModel.nHeadKv());
                assertEquals((long) ffmModelCls.getMethod("nParams").invoke(ffmModel), jniModel.nParams());
                assertEquals((long) ffmModelCls.getMethod("modelSize").invoke(ffmModel), jniModel.modelSize());
                assertEquals((String) ffmModelCls.getMethod("desc").invoke(ffmModel), jniModel.desc());
                assertEquals((Map<String, String>) ffmModelCls.getMethod("getMetadataMap").invoke(ffmModel), jniModel.getMetadataMap());

                String prompt = "Differential verification across Panama FFM and JNI 🚀🔥";
                int[] ffmTokens = (int[]) ffmModelCls.getMethod("tokenize", String.class, boolean.class).invoke(ffmModel, prompt, true);
                int[] jniTokens = jniModel.tokenize(prompt, true);
                assertArrayEquals(ffmTokens, jniTokens, "FFM and JNI tokenization must be bit-identical");

                Method ffmTokenToPieceBytes = ffmModelCls.getMethod("tokenToPieceBytes", int.class);
                Method ffmTokenToPiece = ffmModelCls.getMethod("tokenToPiece", int.class);
                for (int tok : jniTokens) {
                    assertArrayEquals((byte[]) ffmTokenToPieceBytes.invoke(ffmModel, tok), jniModel.tokenToPieceBytes(tok));
                    assertEquals((String) ffmTokenToPiece.invoke(ffmModel, tok), jniModel.tokenToPiece(tok));
                }

                // 3. Deterministic greedy generation parity
                Object ffmGenBuilder = ffmConfigBuilderCls.getConstructor(int.class).newInstance(128);
                ffmConfigBuilderCls.getMethod("cpuThreads", int.class).invoke(ffmGenBuilder, 2);
                Object ffmGenConfig = ffmConfigBuilderCls.getMethod("build").invoke(ffmGenBuilder);

                ArgusContextConfig jniGenConfig = new ArgusContextConfig.Builder(128).cpuThreads(2).build();
                ArgusSamplerConfig greedy = ArgusSamplerConfig.greedy();

                Method ffmCtxInit = ffmContextCls.getMethod("init", ffmModelCls, ffmConfigCls);
                Method ffmGenerate = ffmContextCls.getMethod("generate", String.class, int.class, ArgusSamplerConfig.class, Consumer.class);

                try (AutoCloseable ffmCtx = (AutoCloseable) ffmCtxInit.invoke(null, ffmModel, ffmGenConfig);
                     ArgusContext jniCtx = ArgusContext.init(jniModel, jniGenConfig)) {
                    String ffmOut = (String) ffmGenerate.invoke(ffmCtx, "Hello world", 4, greedy, null);
                    String jniOut = jniCtx.generate("Hello world", 4, greedy, null);
                    assertEquals(ffmOut, jniOut, "FFM and JNI greedy generation must produce identical text");
                }

                // 4. Deterministic embedding extraction parity
                Object ffmEmbdBuilder = ffmConfigBuilderCls.getConstructor(int.class).newInstance(128);
                ffmConfigBuilderCls.getMethod("cpuThreads", int.class).invoke(ffmEmbdBuilder, 2);
                ffmConfigBuilderCls.getMethod("embeddings", boolean.class).invoke(ffmEmbdBuilder, true);
                Object ffmEmbdConfig = ffmConfigBuilderCls.getMethod("build").invoke(ffmEmbdBuilder);

                ArgusContextConfig jniEmbdConfig = new ArgusContextConfig.Builder(128).cpuThreads(2).embeddings(true).build();
                Method ffmExtractEmbeddings = ffmContextCls.getMethod("extractEmbeddings", String.class);

                try (AutoCloseable ffmCtx = (AutoCloseable) ffmCtxInit.invoke(null, ffmModel, ffmEmbdConfig);
                     ArgusContext jniCtx = ArgusContext.init(jniModel, jniEmbdConfig)) {
                    float[] ffmEmbd = (float[]) ffmExtractEmbeddings.invoke(ffmCtx, "Vector parity check");
                    float[] jniEmbd = jniCtx.extractEmbeddings("Vector parity check");
                    assertArrayEquals(ffmEmbd, jniEmbd, 1e-6f, "FFM and JNI embeddings must match within float tolerance");
                }
            }
        }
    }
}
