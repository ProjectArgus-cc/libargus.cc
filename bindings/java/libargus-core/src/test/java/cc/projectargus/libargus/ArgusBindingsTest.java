package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Paths;
import java.util.List;

public class ArgusBindingsTest {

    @Test
    public void testBackendLifecycle() {
        System.out.println("[Java Test] Starting FFM backend lifecycle validation...");

        // 1. Initialize backends
        boolean initStatus = ArgusBackend.init();
        assertTrue(initStatus, "Global backend initialization failed");

        // 2. Query counts and names
        int count = ArgusBackend.getCount();
        assertTrue(count > 0, "No backends detected by unmanaged registry");
        System.out.println("[Java Test] Detected " + count + " native backend(s):");

        List<String> backends = ArgusBackend.getAvailableBackends();
        assertEquals(count, backends.size());
        for (int i = 0; i < count; i++) {
            String name = backends.get(i);
            assertNotNull(name);
            System.out.println("  - Backend " + i + ": " + name);
        }

        // 3. Free backends
        ArgusBackend.free();
        System.out.println("[Java Test] FFM backend lifecycle successfully completed.");
    }

    @Test
    public void testMultimodalWrappers() {
        System.out.println("[Java Test] Starting FFM multimodal wrappers validation...");

        try (Arena arena = Arena.ofConfined()) {
            assertThrows(RuntimeException.class, () -> {
                ArgusMultimodalContext.init(arena, null, Paths.get("non-existent-mmproj.gguf"), 4, false);
            });

            assertThrows(NullPointerException.class, () -> {
                ArgusBitmap.fromRgb(10, 10, null);
            });

            assertThrows(NullPointerException.class, () -> {
                ArgusBitmap.fromPcm(null, 0);
            });

            try (ArgusInputChunks chunks = ArgusInputChunks.init()) {
                assertNotNull(chunks.getHandle());
                assertFalse(chunks.getHandle().equals(Arena.ofConfined().allocate(0)));
            }
        }
        System.out.println("[Java Test] FFM multimodal wrappers successfully validated.");
    }

    @Test
    public void testModelRefCountProtection() {
        System.out.println("[Java Test] Validating model reference counting...");
        ArgusBackend.init();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dummyPtr = arena.allocate(16);
            ArgusModel model = new ArgusModel(dummyPtr);
            assertEquals(1, model.getRefCount());

            model.acquire();
            assertEquals(2, model.getRefCount());

            model.close(); // decrements ref count to 1, native model is NOT closed
            assertEquals(1, model.getRefCount());
            assertNotEquals(MemorySegment.NULL, model.getHandle());

            model.clearHandleForTesting();
            model.close(); // decrements to 0, native model IS closed
            assertEquals(0, model.getRefCount());
            assertEquals(MemorySegment.NULL, model.getHandle());
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testContextInitExceptionSafety() {
        System.out.println("[Java Test] Validating context init exception safety...");
        ArgusBackend.init();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dummyPtr = arena.allocate(16);
            ArgusModel model = new ArgusModel(dummyPtr);
            assertEquals(1, model.getRefCount());

            ArgusContextConfig config = ArgusContextConfig.createDefault(1024);
            assertThrows(RuntimeException.class, () -> {
                // This will fail in native init on a dummy model handle
                ArgusContext.init(arena, model, config);
            });

            // The exception handler must have safely cleaned up the acquired reference!
            assertEquals(1, model.getRefCount());
            model.clearHandleForTesting();
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testReusableVideoItemLoop() {
        System.out.println("[Java Test] Validating reusable ArgusVideoItem loop...");
        boolean initStatus = ArgusBackend.init();
        assertTrue(initStatus);

        try (Arena arena = Arena.ofConfined()) {
            try (ArgusVideoItem item = new ArgusVideoItem()) {
                assertNull(item.bitmap());
                assertNull(item.text());

                // Update item with timestamp text only (no bitmap)
                item.update(MemorySegment.NULL, "[00m10s]");
                assertNull(item.bitmap());
                assertEquals("[00m10s]", item.text());

                // Create a real native bitmap using ArgusBitmap.fromRgb
                MemorySegment rgbData = arena.allocate(3); // 1 pixel
                ArgusBitmap realBitmap1 = ArgusBitmap.fromRgb(1, 1, rgbData);
                MemorySegment handle1 = realBitmap1.getHandle();

                // Pass the handle to item. Note that realBitmap1 must NOT be closed manually,
                // because item will take ownership of this handle and close it when updated/closed.
                item.update(handle1, null);
                assertNotNull(item.bitmap());
                assertEquals(handle1, item.bitmap().getHandle());
                assertNull(item.text());

                // Create another native bitmap
                ArgusBitmap realBitmap2 = ArgusBitmap.fromRgb(1, 1, rgbData);
                MemorySegment handle2 = realBitmap2.getHandle();

                // Updating item with handle2 will automatically close/free handle1!
                item.update(handle2, "frame 2");
                assertNotNull(item.bitmap());
                assertEquals(handle2, item.bitmap().getHandle());
                assertEquals("frame 2", item.text());
            } // item.close() will automatically close/free handle2!
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testContextConfigBuilder() {
        System.out.println("[Java Test] Validating ArgusContextConfig.Builder...");
        
        // Verify Panama struct layout byteSize matches C sizeof(argus_context_params_t) exactly (40 bytes)
        assertEquals(40L, cc.projectargus.libargus.internal.ArgusLayouts.CONTEXT_PARAMS.byteSize());
        assertEquals(32L, cc.projectargus.libargus.internal.ArgusLayouts.CONTEXT_PARAMS.byteOffset(java.lang.foreign.MemoryLayout.PathElement.groupElement("n_seq_max")));
        assertEquals(38L, cc.projectargus.libargus.internal.ArgusLayouts.CONTEXT_PARAMS.byteOffset(java.lang.foreign.MemoryLayout.PathElement.groupElement("kv_unified")));

        // Test default constructor
        ArgusContextConfig.Builder builder1 = new ArgusContextConfig.Builder();
        ArgusContextConfig config1 = builder1.build();
        assertNull(config1.draftModel());
        assertEquals(2048, config1.contextLength());
        assertEquals(ArgusContextConfig.KV_TYPE_F16, config1.typeK());
        assertEquals(ArgusContextConfig.KV_TYPE_F16, config1.typeV());
        assertEquals(0, config1.uBatch());
        assertEquals(0, config1.seqMax());
        assertFalse(config1.enableDraftMtp());
        assertFalse(config1.embeddings());
        assertTrue(config1.kvUnified());
        
        // Test custom constructor and fluent setters
        ArgusContextConfig.Builder builder2 = new ArgusContextConfig.Builder(1024)
            .cpuThreads(4)
            .typeK(ArgusContextConfig.KV_TYPE_Q4_0)
            .typeV(ArgusContextConfig.KV_TYPE_Q8_0)
            .specDraftNMax(5)
            .uBatch(1024)
            .seqMax(2)
            .enableDraftMtp(true)
            .embeddings(true)
            .kvUnified(false);
            
        ArgusContextConfig config2 = builder2.build();
        assertEquals(1024, config2.contextLength());
        assertEquals(4, config2.cpuThreads());
        assertEquals(ArgusContextConfig.KV_TYPE_Q4_0, config2.typeK());
        assertEquals(ArgusContextConfig.KV_TYPE_Q8_0, config2.typeV());
        assertEquals(5, config2.specDraftNMax());
        assertEquals(1024, config2.uBatch());
        assertEquals(2, config2.seqMax());
        assertTrue(config2.enableDraftMtp());
        assertTrue(config2.embeddings());
        assertFalse(config2.kvUnified());
    }

    @Test
    public void testModelVocabAndMetadataNullChecks() {
        System.out.println("[Java Test] Validating vocab and metadata null/error pathways on dummy model...");
        ArgusBackend.init();
        try {
            ArgusModel dummyModel = new ArgusModel(MemorySegment.NULL);
            assertEquals(-1, dummyModel.vocabBos());
            assertEquals(-1, dummyModel.vocabEos());
            assertEquals(-1, dummyModel.vocabEot());
            assertEquals(-1, dummyModel.vocabPad());
            assertEquals(-1, dummyModel.vocabNTokens());
            assertFalse(dummyModel.vocabIsEog(0));
            
            assertEquals(-1, dummyModel.nEmbd());
            assertEquals(-1, dummyModel.nCtxTrain());
            assertEquals(-1, dummyModel.nLayer());
            assertEquals(-1, dummyModel.nHead());
            assertEquals(-1, dummyModel.nHeadKv());
            assertEquals(0L, dummyModel.nParams());
            assertFalse(dummyModel.hasEncoder());
            
            assertNull(dummyModel.getMetadataValue("some_key"));
            assertTrue(dummyModel.getMetadataMap().isEmpty());
            
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment keySeg = arena.allocateFrom("some_key");
                MemorySegment valSeg = arena.allocate(10);
                int res = dummyModel.getMetadataValue(keySeg, valSeg, 10);
                assertEquals(-1, res);
            }
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testSampleTokenWithBiasNullChecks() {
        System.out.println("[Java Test] Validating sampleTokenWithBias null/error pathways...");
        ArgusBackend.init();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dummyBiases = arena.allocate(cc.projectargus.libargus.internal.ArgusLayouts.LOGIT_BIAS, 3);
            int token = (int) ArgusBindings.argus_sample_token_with_bias.invokeExact(
                MemorySegment.NULL, 0, 0.0f, 0.0f, dummyBiases, 3
            );
            assertEquals(-1, token);
        } catch (Throwable t) {
            fail("Exception thrown in native downcall: " + t.getMessage());
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testStructLayoutSizes() {
        System.out.println("[Java Test] Validating FFM struct layout byte sizes against 64-bit alignment...");
        assertEquals(56L, cc.projectargus.libargus.internal.ArgusLayouts.SAMPLER_PARAMS.byteSize(), "SAMPLER_PARAMS layout must be exactly 56 bytes");
        assertEquals(40L, cc.projectargus.libargus.internal.ArgusLayouts.CONTEXT_PARAMS.byteSize(), "CONTEXT_PARAMS layout must be exactly 40 bytes");
        assertEquals(16L, cc.projectargus.libargus.internal.ArgusLayouts.AUDIO_PARAMS.byteSize(), "AUDIO_PARAMS layout must be exactly 16 bytes");
        assertEquals(32L, cc.projectargus.libargus.internal.ArgusLayouts.TOKEN_BATCH.byteSize(), "TOKEN_BATCH layout must be exactly 32 bytes");
        assertEquals(16L, cc.projectargus.libargus.internal.ArgusLayouts.MULTIMODAL_PARAMS.byteSize(), "MULTIMODAL_PARAMS layout must be exactly 16 bytes");
        assertEquals(8L, cc.projectargus.libargus.internal.ArgusLayouts.LOGIT_BIAS.byteSize(), "LOGIT_BIAS layout must be exactly 8 bytes");
    }

    @Test
    public void testCustomNativeDirExtraction() {
        System.out.println("[Java Test] Validating custom native dir property...");
        assertNotNull(ArgusBindings.EXTRACTED_DIR);
        System.out.println("[Java Test] EXTRACTED_DIR was resolved to: " + ArgusBindings.EXTRACTED_DIR);
    }

    @Test
    public void testLibraryVersionAssertion() {
        System.out.println("[Java Test] Validating compiled native library version...");
        assertEquals("1.6.4", ArgusBindings.VERSION);
        try {
            MemorySegment verPtr = (MemorySegment) ArgusBindings.argus_version.invokeExact();
            assertNotNull(verPtr);
            assertFalse(verPtr.equals(MemorySegment.NULL));
            String nativeVer = verPtr.reinterpret(Long.MAX_VALUE).getString(0);
            assertEquals("1.6.4", nativeVer);
            System.out.println("[Java Test] Java static version matches native compiled version: " + nativeVer);
        } catch (Throwable t) {
            fail("Failed to verify native version: " + t.getMessage());
        }
    }

    @Test
    public void testQuantizationTypeIntrospection() {
        System.out.println("[Java Test] Validating GGML quantization type introspection via Java Panama...");
        ArgusBackend.init();
        try {
            assertEquals(2L, ArgusModel.quantTypeSize(ArgusContextConfig.KV_TYPE_F16));
            assertEquals(1, ArgusModel.quantBlockSize(ArgusContextConfig.KV_TYPE_F16));

            assertTrue(ArgusModel.quantTypeSize(ArgusContextConfig.KV_TYPE_Q8_0) > 0);
            assertTrue(ArgusModel.quantBlockSize(ArgusContextConfig.KV_TYPE_Q8_0) > 0);

            assertTrue(ArgusModel.quantTypeSize(ArgusContextConfig.KV_TYPE_Q4_0) > 0);
            assertTrue(ArgusModel.quantBlockSize(ArgusContextConfig.KV_TYPE_Q4_0) > 0);

            assertEquals(0L, ArgusModel.quantTypeSize(-1));
            assertEquals(0, ArgusModel.quantBlockSize(-1));
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testModelMemoryCalculationNullSafety() {
        System.out.println("[Java Test] Validating Model memory calculation null handling downcalls...");
        ArgusBackend.init();
        try {
            long size = (long) ArgusBindings.argus_model_size.invokeExact(MemorySegment.NULL);
            assertEquals(0L, size);

            long kvBytes = (long) ArgusBindings.argus_model_kv_bytes_per_token.invokeExact(
                MemorySegment.NULL, ArgusContextConfig.KV_TYPE_F16, ArgusContextConfig.KV_TYPE_F16
            );
            assertEquals(-1L, kvBytes);

            long estVram = (long) ArgusBindings.argus_model_estimate_vram_bytes.invokeExact(
                MemorySegment.NULL, 4096, ArgusContextConfig.KV_TYPE_Q4_0, ArgusContextConfig.KV_TYPE_Q4_0
            );
            assertEquals(-1L, estVram);
        } catch (Throwable t) {
            fail("Exception thrown in model memory calculation downcalls: " + t.getMessage());
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testThreadControlNullChecks() {
        System.out.println("[Java Test] Validating thread control null/error downcalls...");
        ArgusBackend.init();
        try {
            ArgusBindings.argus_set_n_threads.invokeExact(MemorySegment.NULL, 4, 4);
            int nThreads = (int) ArgusBindings.argus_get_n_threads.invokeExact(MemorySegment.NULL);
            assertEquals(-1, nThreads);

            int nThreadsBatch = (int) ArgusBindings.argus_get_n_threads_batch.invokeExact(MemorySegment.NULL);
            assertEquals(-1, nThreadsBatch);

            ArgusBindings.argus_audio_set_n_threads.invokeExact(MemorySegment.NULL, 4);
            int audioThreads = (int) ArgusBindings.argus_audio_get_n_threads.invokeExact(MemorySegment.NULL);
            assertEquals(-1, audioThreads);
        } catch (Throwable t) {
            fail("Exception thrown in thread control downcalls: " + t.getMessage());
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testMRoPEAndKVCachePositionNullSafety() {
        System.out.println("[Java Test] Validating M-RoPE shape queries and KV cache position downcalls on null segments...");
        ArgusBackend.init();
        try {
            int nPos = (int) ArgusBindings.argus_model_n_pos_per_embd.invokeExact(MemorySegment.NULL);
            assertEquals(-1, nPos);

            boolean isMRoPE = (boolean) ArgusBindings.argus_model_is_mrope.invokeExact(MemorySegment.NULL);
            assertFalse(isMRoPE);

            int posMax = (int) ArgusBindings.argus_kv_cache_seq_pos_max.invokeExact(MemorySegment.NULL, 0);
            assertEquals(-1, posMax);

            int posMin = (int) ArgusBindings.argus_kv_cache_seq_pos_min.invokeExact(MemorySegment.NULL, 0);
            assertEquals(-1, posMin);
        } catch (Throwable t) {
            fail("Exception thrown in M-RoPE/KV cache position downcalls: " + t.getMessage());
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    public void testSamplerParamsLayoutAndConfig() {
        System.out.println("[Java Test] Validating SamplerParams struct layout and ArgusSamplerConfig...");
        assertEquals(56L, cc.projectargus.libargus.internal.ArgusLayouts.SAMPLER_PARAMS.byteSize());

        ArgusSamplerConfig defaultCfg = ArgusSamplerConfig.createDefault();
        assertEquals(0.7f, defaultCfg.temperature());
        assertEquals(1.1f, defaultCfg.repeatPenalty());
        assertEquals(64, defaultCfg.repeatLastN());
        assertEquals(0.90f, defaultCfg.topP());
        assertEquals(0.05f, defaultCfg.minP());
        assertEquals(40, defaultCfg.topK());
        assertEquals(0.0f, defaultCfg.dryMultiplier());
        assertEquals(-1L, defaultCfg.seed());

        ArgusSamplerConfig customCfg = new ArgusSamplerConfig.Builder()
            .temperature(0.5f)
            .repeatPenalty(1.2f)
            .repeatLastN(32)
            .frequencyPenalty(0.1f)
            .presencePenalty(0.2f)
            .topP(0.85f)
            .minP(0.10f)
            .topK(20)
            .seed(987654321)
            .dry(0.5f, 1.8f, 3, 100)
            .build();

        assertEquals(0.5f, customCfg.temperature());
        assertEquals(1.2f, customCfg.repeatPenalty());
        assertEquals(32, customCfg.repeatLastN());
        assertEquals(0.1f, customCfg.frequencyPenalty());
        assertEquals(0.2f, customCfg.presencePenalty());
        assertEquals(0.85f, customCfg.topP());
        assertEquals(0.10f, customCfg.minP());
        assertEquals(20, customCfg.topK());
        assertEquals(987654321, customCfg.seed());
        assertEquals(0.5f, customCfg.dryMultiplier());
        assertEquals(1.8f, customCfg.dryBase());
        assertEquals(3, customCfg.dryAllowedLength());
        assertEquals(100, customCfg.dryPenaltyLastN());

        ArgusSamplerConfig greedyCfg = ArgusSamplerConfig.greedy();
        assertEquals(0.0f, greedyCfg.temperature());
    }

    @Test
    public void testFangedEndToEndExecutionWithTinyModel() {
        System.out.println("[Java Test] Starting fanged end-to-end model execution verification in Java FFM...");
        java.nio.file.Path root = java.nio.file.Paths.get("").toAbsolutePath();
        while (root != null && !java.nio.file.Files.exists(root.resolve("tests/data/tiny.gguf"))) {
            root = root.getParent();
        }
        assertNotNull(root, "Could not locate project root containing tests/data/tiny.gguf");
        java.nio.file.Path modelPath = root.resolve("tests/data/tiny.gguf");
        assertTrue(java.nio.file.Files.exists(modelPath), "tests/data/tiny.gguf missing at " + modelPath);

        boolean initStatus = ArgusBackend.init();
        assertTrue(initStatus);

        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, modelPath, 0, false);
             ArgusModel draftModel = ArgusModel.load(arena, modelPath, 0, false)) {

            // 1. Assert model metadata and shape introspection
            assertEquals(1, model.vocabBos());
            assertEquals(2, model.vocabEos());
            assertEquals(3, model.vocabPad());
            assertEquals(64, model.vocabNTokens());
            assertEquals(32, model.nEmbd());
            assertEquals(512, model.nCtxTrain());
            assertEquals(1, model.nLayer());
            assertEquals(2, model.nHead());
            assertEquals(2, model.nHeadKv());

            // 2. Initialize context with speculative draft model
            ArgusContextConfig config = new ArgusContextConfig.Builder(256)
                .draftModel(draftModel)
                .cpuThreads(2)
                .specDraftNMax(4)
                .seqMax(2)
                .kvUnified(true)
                .build();

            try (ArgusContext context = ArgusContext.init(arena, model, config)) {
                assertTrue(context.hasDraftContext(), "Speculative draft context should be active");

                // 3. Batch decode initial prompt
                int[] promptTokens = new int[] { 1, 4, 5, 6, 7 }; // 5 tokens (BOS, a, b, c, d)
                MemorySegment promptSeg = arena.allocate(ValueLayout.JAVA_INT, promptTokens.length);
                for (int i = 0; i < promptTokens.length; i++) {
                    promptSeg.setAtIndex(ValueLayout.JAVA_INT, i, promptTokens[i]);
                }

                int decRes1 = context.decodeBatch(promptSeg, promptTokens.length, 0, 0, true);
                assertEquals(0, decRes1);
                assertEquals(4, context.getSeqPosMax(0));
                System.out.println("  - Java batch decoded. Initial seq_pos_max: " + context.getSeqPosMax(0));

                // 4. Automagic rollback verification
                int[] branchTokens = new int[] { 8, 9 }; // e, f starting at pos 2
                MemorySegment branchSeg = arena.allocate(ValueLayout.JAVA_INT, branchTokens.length);
                for (int i = 0; i < branchTokens.length; i++) {
                    branchSeg.setAtIndex(ValueLayout.JAVA_INT, i, branchTokens[i]);
                }

                int decRes2 = context.decodeBatch(branchSeg, branchTokens.length, 2, 0, true);
                assertEquals(0, decRes2);
                assertEquals(3, context.getSeqPosMax(0));
                System.out.println("  - Java prefix rollback verified in lockstep. New seq_pos_max: " + context.getSeqPosMax(0));

                // 5. Extended sampling verification (Deterministic Seeding)
                ArgusSamplerConfig sampleCfg = new ArgusSamplerConfig.Builder()
                    .temperature(0.7f)
                    .seed(42)
                    .build();
                int sampledToken = context.sampleToken(0, sampleCfg);
                assertTrue(sampledToken >= 0 && sampledToken < 64);
                System.out.println("  - Java extended sampler produced token: " + sampledToken);

                // Verify logits consumption: immediate subsequent sample without decode returns -2
                int reSample = context.sampleToken(0, sampleCfg);
                assertEquals(-2, reSample);
                System.out.println("  - Java logits consumption verified (returned -2 on repeated sample).");

                // 6. Multi-Sequence Logits Isolation Verification
                int[] s0Toks = new int[] { 1, 10 };
                MemorySegment s0Seg = arena.allocate(ValueLayout.JAVA_INT, s0Toks.length);
                for (int i = 0; i < s0Toks.length; i++) s0Seg.setAtIndex(ValueLayout.JAVA_INT, i, s0Toks[i]);
                assertEquals(0, context.decodeBatch(s0Seg, s0Toks.length, 0, 0, true));

                int[] s1Toks = new int[] { 1, 20 };
                MemorySegment s1Seg = arena.allocate(ValueLayout.JAVA_INT, s1Toks.length);
                for (int i = 0; i < s1Toks.length; i++) s1Seg.setAtIndex(ValueLayout.JAVA_INT, i, s1Toks[i]);
                assertEquals(0, context.decodeBatch(s1Seg, s1Toks.length, 0, 1, true));

                // Sampling seq 0 must return -2 because seq 1 was decoded last
                assertEquals(-2, context.sampleToken(0, sampleCfg));
                System.out.println("  - Java cross-sequence logits contamination prevented (-2 returned for seq 0).");

                // Sampling seq 1 succeeds
                int s1Token = context.sampleToken(1, sampleCfg);
                assertTrue(s1Token >= 0 && s1Token < 64);
                System.out.println("  - Java seq 1 sampled successfully: " + s1Token);

                // 7. Stale Logits Invalidation on Partial KV Cache Mutation
                assertEquals(0, context.decodeBatch(s0Seg, s0Toks.length, 0, 0, true));
                context.clearCacheSlot(0, 1, -1);
                int invalidatedSample = context.sampleToken(0, sampleCfg);
                assertEquals(-2, invalidatedSample, "Partial KV mutation must invalidate pending logits (-2)");
                System.out.println("  - Java stale logits invalidation verified upon KV cache mutation.");

                // 8. Logit steering bias verification
                int[] steerToks = new int[] { 1, 4 };
                MemorySegment steerSeg = arena.allocate(ValueLayout.JAVA_INT, steerToks.length);
                for (int i = 0; i < steerToks.length; i++) steerSeg.setAtIndex(ValueLayout.JAVA_INT, i, steerToks[i]);
                assertEquals(0, context.decodeBatch(steerSeg, steerToks.length, 0, 0, true));

                MemorySegment biasSeg1 = arena.allocate(cc.projectargus.libargus.internal.ArgusLayouts.LOGIT_BIAS, 1);
                biasSeg1.setAtIndex(ValueLayout.JAVA_INT, 0, 10);
                biasSeg1.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 1000.0f);

                int biasedToken1 = context.sampleTokenWithBias(0, sampleCfg, biasSeg1, 1);
                assertEquals(10, biasedToken1);
                System.out.println("  - Java logit bias steering produced forced token: " + biasedToken1);

                assertEquals(0, context.decodeBatch(steerSeg, steerToks.length, 0, 0, true));
                MemorySegment biasSeg2 = arena.allocate(cc.projectargus.libargus.internal.ArgusLayouts.LOGIT_BIAS, 1);
                biasSeg2.setAtIndex(ValueLayout.JAVA_INT, 0, 15);
                biasSeg2.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 1000.0f);

                int biasedToken2 = context.sampleTokenWithBias(0, sampleCfg, biasSeg2, 1);
                assertEquals(15, biasedToken2);
                System.out.println("  - Java logit bias steering dynamically updated to token: " + biasedToken2);

                // 9. Stochastic Distribution Sampling & Entropy Divergence
                java.util.Set<Integer> distinctToks = new java.util.HashSet<>();
                for (int seed = 1; seed <= 10; seed++) {
                    assertEquals(0, context.decodeBatch(steerSeg, steerToks.length, 0, 0, true));
                    ArgusSamplerConfig distCfg = new ArgusSamplerConfig.Builder()
                        .temperature(1.5f)
                        .topK(50)
                        .topP(1.0f)
                        .seed(seed)
                        .build();
                    int t = context.sampleToken(0, distCfg);
                    assertTrue(t >= 0);
                    distinctToks.add(t);
                }
                assertTrue(distinctToks.size() >= 2, "High-temperature distribution sampling across seeds must produce divergent tokens");
                System.out.println("  - Java stochastic distribution entropy verified: " + distinctToks.size() + " distinct tokens.");

                // 10. Deterministic Seeding vs Entropy Verification
                java.util.List<Integer> run1 = new java.util.ArrayList<>();
                java.util.List<Integer> run2 = new java.util.ArrayList<>();
                ArgusSamplerConfig seedCfg = new ArgusSamplerConfig.Builder()
                    .temperature(0.8f)
                    .topK(10)
                    .seed(777)
                    .build();

                // Run 1
                context.clearCacheSlot(0, 0, -1);
                context.resetSampler(0);
                assertEquals(0, context.getSamplerHistoryCount(0));
                assertFalse(context.hasSamplerPending(0));

                int[] seedPrompt = new int[] { 1, 4, 5 };
                MemorySegment spSeg = arena.allocate(ValueLayout.JAVA_INT, seedPrompt.length);
                for (int i = 0; i < seedPrompt.length; i++) spSeg.setAtIndex(ValueLayout.JAVA_INT, i, seedPrompt[i]);
                assertEquals(0, context.decodeBatch(spSeg, seedPrompt.length, 0, 0, true));

                for (int i = 0; i < 4; i++) {
                    assertFalse(context.hasSamplerPending(0));
                    assertEquals(i, context.getSamplerHistoryCount(0));

                    int t = context.sampleToken(0, seedCfg);
                    assertTrue(t >= 0);
                    assertTrue(context.hasSamplerPending(0));
                    assertEquals(i, context.getSamplerHistoryCount(0));

                    run1.add(t);
                    MemorySegment nextSeg = arena.allocate(ValueLayout.JAVA_INT, 1);
                    nextSeg.setAtIndex(ValueLayout.JAVA_INT, 0, t);
                    assertEquals(0, context.decodeBatch(nextSeg, 1, 3 + i, 0, true));

                    assertFalse(context.hasSamplerPending(0));
                    assertEquals(i + 1, context.getSamplerHistoryCount(0));
                }
                assertEquals(4, context.getSamplerHistoryCount(0));

                // Run 2 (same seed 777)
                context.clearCacheSlot(0, 0, -1);
                context.resetSampler(0);
                assertEquals(0, context.decodeBatch(spSeg, seedPrompt.length, 0, 0, true));

                for (int i = 0; i < 4; i++) {
                    int t = context.sampleToken(0, seedCfg);
                    assertTrue(t >= 0);
                    run2.add(t);
                    MemorySegment nextSeg = arena.allocate(ValueLayout.JAVA_INT, 1);
                    nextSeg.setAtIndex(ValueLayout.JAVA_INT, 0, t);
                    assertEquals(0, context.decodeBatch(nextSeg, 1, 3 + i, 0, true));
                }

                assertEquals(run1, run2, "Seed 777 should produce deterministic token stream across runs");
                System.out.println("  - Java seed reproducibility & monotonic history growth verified: " + run1);

                // 11. Mismatched Pending Sample Rejection
                context.clearCacheSlot(0, 0, -1);
                context.resetSampler(0);
                assertEquals(0, context.decodeBatch(spSeg, seedPrompt.length, 0, 0, true));
                int sampT = context.sampleToken(0, seedCfg);
                assertTrue(sampT >= 0);
                assertTrue(context.hasSamplerPending(0));
                assertEquals(0, context.getSamplerHistoryCount(0));

                int mismatchT = (sampT + 1) % 64;
                MemorySegment mmSeg = arena.allocate(ValueLayout.JAVA_INT, 1);
                mmSeg.setAtIndex(ValueLayout.JAVA_INT, 0, mismatchT);
                assertEquals(0, context.decodeBatch(mmSeg, 1, 3, 0, true));
                assertFalse(context.hasSamplerPending(0));
                assertEquals(1, context.getSamplerHistoryCount(0));
                System.out.println("  - Java mismatched pending sample reconciliation verified.");

                // 12. Decoupled Priming Persistence Across KV Rollback
                context.clearCacheSlot(0, 0, -1);
                context.resetSampler(0);
                int[] primedToks = new int[] { 10, 11, 12, 13, 14, 15 };
                MemorySegment primedSeg = arena.allocate(ValueLayout.JAVA_INT, primedToks.length);
                for (int i = 0; i < primedToks.length; i++) primedSeg.setAtIndex(ValueLayout.JAVA_INT, i, primedToks[i]);
                assertEquals(0, context.primeSampler(0, primedSeg, primedToks.length));
                assertEquals(6, context.getSamplerHistoryCount(0));

                int[] promptD = new int[] { 1, 4, 5, 6 };
                MemorySegment pDSeg = arena.allocate(ValueLayout.JAVA_INT, promptD.length);
                for (int i = 0; i < promptD.length; i++) pDSeg.setAtIndex(ValueLayout.JAVA_INT, i, promptD[i]);
                assertEquals(0, context.decodeBatch(pDSeg, promptD.length, 0, 0, true));
                assertEquals(6, context.getSamplerHistoryCount(0));

                int sTok1 = context.sampleToken(0, sampleCfg);
                assertTrue(sTok1 >= 0);
                assertTrue(context.hasSamplerPending(0));
                MemorySegment c1Seg = arena.allocate(ValueLayout.JAVA_INT, 1);
                c1Seg.setAtIndex(ValueLayout.JAVA_INT, 0, sTok1);
                assertEquals(0, context.decodeBatch(c1Seg, 1, 4, 0, true));
                assertFalse(context.hasSamplerPending(0));
                assertEquals(7, context.getSamplerHistoryCount(0));

                int sTok2 = context.sampleToken(0, sampleCfg);
                assertTrue(sTok2 >= 0);
                assertTrue(context.hasSamplerPending(0));
                MemorySegment c2Seg = arena.allocate(ValueLayout.JAVA_INT, 1);
                c2Seg.setAtIndex(ValueLayout.JAVA_INT, 0, sTok2);
                assertEquals(0, context.decodeBatch(c2Seg, 1, 5, 0, true));
                assertFalse(context.hasSamplerPending(0));
                assertEquals(8, context.getSamplerHistoryCount(0));

                int[] branchD = new int[] { 20, 21 };
                MemorySegment brSeg = arena.allocate(ValueLayout.JAVA_INT, branchD.length);
                for (int i = 0; i < branchD.length; i++) brSeg.setAtIndex(ValueLayout.JAVA_INT, i, branchD[i]);
                assertEquals(0, context.decodeBatch(brSeg, branchD.length, 2, 0, true));
                assertEquals(6, context.getSamplerHistoryCount(0));
                System.out.println("  - Java coordinate-decoupled rollback verified (primed tokens preserved across KV rollback).");

                // 13. Sampler Priming & Lifecycle
                int[] primeToks = new int[] { 12, 13, 14 };
                MemorySegment primeSeg = arena.allocate(ValueLayout.JAVA_INT, primeToks.length);
                for (int i = 0; i < primeToks.length; i++) primeSeg.setAtIndex(ValueLayout.JAVA_INT, i, primeToks[i]);
                assertEquals(0, context.primeSampler(0, primeSeg, 3));
                assertEquals(9, context.getSamplerHistoryCount(0));
                context.truncateSampler(0, 1);
                assertEquals(1, context.getSamplerHistoryCount(0));
                context.resetSampler(0);
                assertEquals(0, context.getSamplerHistoryCount(0));
                System.out.println("  - Java sampler lifecycle (prime, truncate, reset) verified.");
            }
        } finally {
            ArgusBackend.free();
        }
    }
}

