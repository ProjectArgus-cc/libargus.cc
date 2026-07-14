package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

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
        
        // Test default constructor
        ArgusContextConfig.Builder builder1 = new ArgusContextConfig.Builder();
        ArgusContextConfig config1 = builder1.build();
        assertNull(config1.draftModel());
        assertEquals(2048, config1.contextLength());
        assertEquals(ArgusContextConfig.KV_TYPE_F16, config1.typeK());
        assertEquals(ArgusContextConfig.KV_TYPE_F16, config1.typeV());
        assertFalse(config1.enableDraftMtp());
        assertFalse(config1.embeddings());
        
        // Test custom constructor and fluent setters
        ArgusContextConfig.Builder builder2 = new ArgusContextConfig.Builder(1024)
            .cpuThreads(4)
            .typeK(ArgusContextConfig.KV_TYPE_Q4_0)
            .typeV(ArgusContextConfig.KV_TYPE_Q8_0)
            .specDraftNMax(5)
            .enableDraftMtp(true)
            .embeddings(true);
            
        ArgusContextConfig config2 = builder2.build();
        assertEquals(1024, config2.contextLength());
        assertEquals(4, config2.cpuThreads());
        assertEquals(ArgusContextConfig.KV_TYPE_Q4_0, config2.typeK());
        assertEquals(ArgusContextConfig.KV_TYPE_Q8_0, config2.typeV());
        assertEquals(5, config2.specDraftNMax());
        assertTrue(config2.enableDraftMtp());
        assertTrue(config2.embeddings());
    }
}
