package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
    public void testCustomNativeDirExtraction() {
        System.out.println("[Java Test] Validating custom native dir property...");
        assertNotNull(ArgusBindings.EXTRACTED_DIR);
        System.out.println("[Java Test] EXTRACTED_DIR was resolved to: " + ArgusBindings.EXTRACTED_DIR);
    }

    @Test
    public void testLibraryVersionAssertion() {
        System.out.println("[Java Test] Validating compiled native library version...");
        assertEquals("1.5.2", ArgusBindings.VERSION);
        try {
            MemorySegment verPtr = (MemorySegment) ArgusBindings.argus_version.invokeExact();
            assertNotNull(verPtr);
            assertFalse(verPtr.equals(MemorySegment.NULL));
            String nativeVer = verPtr.reinterpret(Long.MAX_VALUE).getString(0);
            assertEquals("1.5.2", nativeVer);
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
}

