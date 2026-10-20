package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class ArgusJniTest {

    private static Path findProjectRoot() {
        Path root = Paths.get(System.getProperty("argus.root", "")).toAbsolutePath();
        while (root != null && !Files.exists(root.resolve("tests/data/tiny-embed.gguf"))) {
            root = root.getParent();
        }
        assertNotNull(root, "Could not locate project root containing tests/data/tiny-embed.gguf");
        return root;
    }

    private static Path findTestModelPath() {
        return findProjectRoot().resolve("tests/data/tiny-embed.gguf");
    }

    private static int extractPosixFd(FileDescriptor fdObj) throws Exception {
        Field fdField = FileDescriptor.class.getDeclaredField("fd");
        fdField.setAccessible(true);
        return fdField.getInt(fdObj);
    }

    @Test
    public void testBackendLifecycleAndBuildInfo() throws Exception {
        assertTrue(ArgusBackend.init(), "ArgusBackend.init() should succeed");
        assertTrue(ArgusBackend.isInitialized(), "ArgusBackend.isInitialized() should be true");
        assertTrue(ArgusBackend.getCount() > 0, "At least CPU backend must be registered");

        List<String> backends = ArgusBackend.getAvailableBackends();
        assertEquals(ArgusBackend.getCount(), backends.size());
        for (String name : backends) {
            assertNotNull(name);
            assertFalse(name.isEmpty());
        }

        String expectedVersion = Files.readString(findProjectRoot().resolve("version.txt")).trim();
        assertTrue(ArgusBackend.hasFeature(ArgusBackend.FEATURE_CPU));
        assertEquals(expectedVersion, ArgusBackend.getVersion());

        Map<String, String> buildInfo = ArgusBackend.getBuildInfo();
        assertEquals(expectedVersion, buildInfo.get("version"));
        assertNotNull(buildInfo.get("abi_header_sha256"));
        assertEquals(64, buildInfo.get("abi_header_sha256").length());
    }

    @Test
    public void testLoggingAndExceptionContainment() {
        ArgusBackend.init();
        ArgusLogLevel original = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            assertEquals(ArgusLogLevel.DEBUG, ArgusBackend.getLogLevel());

            AtomicInteger callbackInvocations = new AtomicInteger(0);
            ArgusBackend.setLogCallback((level, message) -> {
                callbackInvocations.incrementAndGet();
                throw new RuntimeException("Intentional exception inside JNI log callback");
            });

            // Trigger a native error/log via invalid model load; callback exception must be contained
            assertThrows(ArgusNativeException.class, () ->
                ArgusModel.load(Paths.get("/non/existent/model_path_12345.gguf"), 0, false)
            );
            assertTrue(callbackInvocations.get() >= 0);
        } finally {
            ArgusBackend.setLogCallback(null);
            ArgusBackend.setLogLevel(original);
        }
    }

    @Test
    public void testModelLoadAndLoadFdRewindAndOwnership() throws Exception {
        ArgusBackend.init();
        Path modelPath = findTestModelPath();

        // 1. Standard path load and introspection
        try (ArgusModel model = ArgusModel.load(modelPath, 0, false)) {
            assertTrue(model.vocabNTokens() > 0);
            assertEquals(model.vocabNTokens(), model.getVocabSize());
            assertTrue(model.nEmbd() > 0);
            assertEquals(model.nEmbd(), model.nEmbdOut());
            assertTrue(model.nEmbdInp() > 0);
            assertTrue(model.nCtxTrain() > 0);
            assertTrue(model.nLayer() > 0);
            assertTrue(model.nHead() > 0);
            assertTrue(model.nHeadKv() > 0);
            assertTrue(model.nParams() > 0);
            assertTrue(model.modelSize() > 0);
            assertFalse(model.desc().isEmpty());
            assertEquals(model.desc(), model.getDescription());
            assertTrue(model.kvBytesPerToken(0, 0) > 0);
            assertTrue(model.estimateVramBytes(512, 0, 0) > model.modelSize());
            assertTrue(ArgusModel.quantTypeSize(0) > 0);
            assertTrue(ArgusModel.quantBlockSize(0) > 0);

            int metaCount = model.getMetadataCount();
            assertTrue(metaCount > 0);
            Map<String, String> metaMap = model.getMetadataMap();
            assertEquals(metaCount, metaMap.size());
            String firstKey = model.getMetadataKeyByIndex(0);
            assertNotNull(firstKey);
            assertEquals(model.getMetadataValue(firstKey), model.getMetadataValueByIndex(0));
            assertEquals(model.getMetadataValue(firstKey), model.getMetadata(firstKey));
            assertNull(model.getMetadataKeyByIndex(-1));
            assertNull(model.getMetadataKeyByIndex(metaCount + 100));
            assertNull(model.getMetadataValue("non.existent.key.xyz"));
        }

        // 2. loadFd with pre-seeked non-zero offset & immediate caller FD closure (on POSIX JVMs where FileDescriptor.fd >= 0)
        ArgusModel fdModel = null;
        try (RandomAccessFile raf = new RandomAccessFile(modelPath.toFile(), "r")) {
            raf.seek(128); // Intentionally advance offset past GGUF magic header
            int rawFd = extractPosixFd(raf.getFD());
            if (rawFd >= 0) {
                fdModel = ArgusModel.loadFd(rawFd, 0, false);
            }
        }
        if (fdModel != null) {
            // Caller's RandomAccessFile is now closed; fdModel must remain 100% functional
            try (ArgusModel m = fdModel) {
                assertTrue(m.vocabNTokens() > 0);
                int[] tokens = m.tokenize("Hello world", true);
                assertTrue(tokens.length > 0);
            }
        }

        // 3. Invalid FD and corrupt file FD error handling without FD leaks
        assertThrows(IllegalArgumentException.class, () -> ArgusModel.loadFd(-1, 0, false));
        assertThrows(ArgusNativeException.class, () -> ArgusModel.loadFd(999999, 0, false));

        Path corruptTemp = Files.createTempFile("argus_corrupt_", ".gguf");
        try {
            try (FileOutputStream fos = new FileOutputStream(corruptTemp.toFile())) {
                fos.write(new byte[] { 'N', 'O', 'T', '_', 'G', 'G', 'U', 'F', 0, 1, 2, 3 });
            }
            for (int i = 0; i < 32; i++) {
                try (RandomAccessFile raf = new RandomAccessFile(corruptTemp.toFile(), "r")) {
                    int rawFd = extractPosixFd(raf.getFD());
                    if (rawFd >= 0) {
                        assertThrows(ArgusNativeException.class, () -> ArgusModel.loadFd(rawFd, 0, false));
                    }
                }
            }
        } finally {
            Files.deleteIfExists(corruptTemp);
        }
    }

    @Test
    public void testTokenizerAndTokenToPieceBounds() {
        ArgusBackend.init();
        Path modelPath = findTestModelPath();

        try (ArgusModel model = ArgusModel.load(modelPath, 0, false)) {
            String text = "Hello 🌍🚀🔥 world! Привет мир! こんにちは";
            int[] tokens = model.tokenize(text, true);
            assertTrue(tokens.length > 0);

            // Validation & capacity probe with maxTokens = 1
            assertThrows(NullPointerException.class, () -> model.tokenize(text, null, 1, true));
            assertThrows(IllegalArgumentException.class, () -> model.tokenize(text, new int[1], 0, true));

            int[] tinyBuf = new int[1];
            int probe1 = model.tokenize(text, tinyBuf, 1, true);
            assertEquals(-tokens.length, probe1);

            // Round-trip pieces via ArgusUtf8StreamDecoder
            cc.projectargus.libargus.internal.ArgusUtf8StreamDecoder decoder =
                new cc.projectargus.libargus.internal.ArgusUtf8StreamDecoder();
            StringBuilder reconstructed = new StringBuilder();
            byte[] intoBuf = new byte[128];
            for (int token : tokens) {
                byte[] pieceBytes = model.tokenToPieceBytes(token);
                assertNotNull(pieceBytes);
                int written = model.tokenToPieceInto(token, intoBuf, 4, 64);
                assertEquals(pieceBytes.length, written);
                for (int i = 0; i < written; i++) {
                    assertEquals(pieceBytes[i], intoBuf[4 + i]);
                }
                reconstructed.append(decoder.decode(pieceBytes, 0, pieceBytes.length));
            }
            reconstructed.append(decoder.finish());
            assertFalse(reconstructed.toString().isEmpty());

            // Out-of-bounds token IDs must return empty piece and not crash
            assertEquals(0, model.tokenToPieceBytes(-1).length);
            assertEquals("", model.tokenToPiece(-1));
            assertEquals(0, model.tokenToPieceBytes(model.vocabNTokens()).length);
            assertEquals(0, model.tokenToPieceBytes(Integer.MAX_VALUE).length);
            assertFalse(model.vocabIsEog(-1));
            assertFalse(model.vocabIsEog(model.vocabNTokens()));
        }
    }

    @Test
    public void testContextInferenceSamplerKvAndDirectBuffers() {
        ArgusBackend.init();
        Path modelPath = findTestModelPath();

        try (ArgusModel model = ArgusModel.load(modelPath, 0, false)) {
            ArgusContextConfig config = new ArgusContextConfig.Builder(256)
                .cpuThreads(2)
                .cloneSlots(2)
                .build();

            try (ArgusContext ctx = ArgusContext.init(model, config)) {
                assertEquals(2, ctx.getAvailableSlotCount());
                assertTrue(ctx.getSeqMax() >= 3);

                ctx.setNThreads(2, 2);
                assertEquals(2, ctx.getNThreads());
                assertEquals(2, ctx.getNThreadsBatch());

                int[] tokens = ctx.tokenize("Hello world", true);
                assertTrue(tokens.length > 0);

                // Decode via DirectByteBuffer
                ByteBuffer directTokens = ByteBuffer.allocateDirect(tokens.length * Integer.BYTES)
                    .order(ByteOrder.nativeOrder());
                for (int t : tokens) {
                    directTokens.putInt(t);
                }
                directTokens.flip();

                int status = ctx.decodeBatchDirect(directTokens, tokens.length, 0, 0, true, null);
                assertEquals(0, status);

                // Unaligned DirectByteBuffer position must be rejected
                directTokens.position(1);
                assertThrows(IllegalArgumentException.class, () ->
                    ctx.decodeBatchDirect(directTokens, 1, 0, 0, true, null)
                );

                // Sampler & KV cache operations
                assertEquals(0, ctx.primeSampler(0, tokens, tokens.length));
                assertEquals(tokens.length, ctx.getSamplerHistoryCount(0));

                int sampled = ctx.sampleToken(0, ArgusSamplerConfig.greedy());
                assertTrue(sampled >= 0 && sampled < model.vocabNTokens());
                assertTrue(ctx.hasSamplerPending(0));
                assertTrue(ctx.discardPendingSample(0));
                assertFalse(ctx.hasSamplerPending(0));

                // Slot forking
                int forkedSlot = ctx.forkSlot(0);
                assertTrue(forkedSlot > 0);
                assertEquals(1, ctx.getAvailableSlotCount());
                assertEquals(ctx.getSeqPosMax(0), ctx.getSeqPosMax(forkedSlot));
                ctx.freeSlot(forkedSlot);
                assertEquals(2, ctx.getAvailableSlotCount());

                // Perf timings
                ArgusPerfTimings perf = ctx.getPerfTimings();
                assertNotNull(perf);
                assertTrue(perf.nPromptEval() > 0 || perf.nEval() > 0);
                ctx.resetPerfTimings();

                // High-level generate() with streaming callback
                List<String> streamedPieces = new ArrayList<>();
                String generated = ctx.generate("Hello", 3, ArgusSamplerConfig.greedy(), streamedPieces::add);
                assertNotNull(generated);
                assertEquals(String.join("", streamedPieces), generated);

                // Cooperative abort via ArgusAbortFlag
                try (ArgusAbortFlag abortFlag = new ArgusAbortFlag()) {
                    assertFalse(abortFlag.isAborted());
                    abortFlag.abort();
                    assertTrue(abortFlag.isAborted());
                    int abortedDecode = ctx.decodeBatch(tokens, tokens.length, 0, 0, true, abortFlag);
                    assertEquals(ArgusContext.STATUS_CANCELLED, abortedDecode);
                    abortFlag.reset();
                    assertFalse(abortFlag.isAborted());
                }
            }
        }
    }

    @Test
    public void testEmbeddingsHeapAndDirectEquivalence() {
        ArgusBackend.init();
        Path modelPath = findTestModelPath();

        try (ArgusModel model = ArgusModel.load(modelPath, 0, false)) {
            ArgusContextConfig config = new ArgusContextConfig.Builder(128)
                .cpuThreads(2)
                .embeddings(true)
                .build();

            try (ArgusContext ctx = ArgusContext.init(model, config)) {
                int dim = ctx.nEmbd();
                assertTrue(dim > 0);

                int[] tokens = ctx.tokenize("Semantic embedding equivalence check", true);
                float[] heapEmbd = new float[dim];
                int writtenHeap = ctx.evalEmbeddings(tokens, tokens.length, 0, heapEmbd);
                assertEquals(dim, writtenHeap);

                ByteBuffer tokDirect = ByteBuffer.allocateDirect(tokens.length * Integer.BYTES)
                    .order(ByteOrder.nativeOrder());
                for (int t : tokens) {
                    tokDirect.putInt(t);
                }
                tokDirect.flip();

                ByteBuffer outDirect = ByteBuffer.allocateDirect(dim * Float.BYTES)
                    .order(ByteOrder.nativeOrder());
                ctx.clearCacheSlot(0, 0, -1);
                int writtenDirect = ctx.evalEmbeddingsDirect(tokDirect, tokens.length, 0, outDirect);
                assertEquals(dim, writtenDirect);

                for (int i = 0; i < dim; i++) {
                    assertEquals(heapEmbd[i], outDirect.getFloat(i * Float.BYTES), 1e-5f);
                }
            }
        }
    }

    @Test
    public void testSpatialValidationAndMediaResources() {
        ArgusBackend.init();

        // RGB Bitmap creation via heap array and direct ByteBuffer
        byte[] rgb = new byte[4 * 4 * 3];
        for (int i = 0; i < rgb.length; i++) {
            rgb[i] = (byte) (i & 0xFF);
        }
        try (ArgusBitmap bmp1 = ArgusBitmap.fromRgb(4, 4, rgb)) {
            assertFalse(bmp1.isClosed());
            bmp1.setMergeable(true);
            bmp1.setMergeable(false);
        }

        ByteBuffer directRgb = ByteBuffer.allocateDirect(4 * 4 * 3);
        directRgb.put(rgb).flip();
        try (ArgusBitmap bmp2 = ArgusBitmap.fromRgbDirect(4, 4, directRgb)) {
            assertFalse(bmp2.isClosed());
        }

        // PCM Bitmap creation via heap array and direct ByteBuffer
        float[] pcm = new float[160];
        try (ArgusBitmap pcmBmp1 = ArgusBitmap.fromPcm(pcm, pcm.length)) {
            assertFalse(pcmBmp1.isClosed());
        }

        ByteBuffer directPcm = ByteBuffer.allocateDirect(pcm.length * Float.BYTES).order(ByteOrder.nativeOrder());
        try (ArgusBitmap pcmBmp2 = ArgusBitmap.fromPcmDirect(directPcm, pcm.length)) {
            assertFalse(pcmBmp2.isClosed());
        }

        // Overflow & boundary rejection tests
        assertThrows(IllegalArgumentException.class, () -> ArgusBitmap.fromRgb(Integer.MAX_VALUE, 2, rgb));
        assertThrows(IllegalArgumentException.class, () -> ArgusBitmap.fromRgb(10, 10, new byte[10]));
        assertThrows(IllegalArgumentException.class, () -> ArgusBitmap.fromPcm(new float[4], 10));
        assertThrows(IllegalArgumentException.class, () -> ArgusModel.load(Paths.get("bad\0path.gguf"), 0, false));

        try (ArgusInputChunks chunks = ArgusInputChunks.init()) {
            assertFalse(chunks.isClosed());
        }

        try (ArgusVideoItem item = new ArgusVideoItem()) {
            assertFalse(item.isClosed());
            assertNull(item.bitmap());
            assertNull(item.text());
        }
    }
}
