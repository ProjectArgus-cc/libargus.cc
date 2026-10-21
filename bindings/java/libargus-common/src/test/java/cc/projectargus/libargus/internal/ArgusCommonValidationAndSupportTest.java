package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusNativeException;
import cc.projectargus.libargus.ArgusSamplerConfig;
import cc.projectargus.libargus.spi.AbstractNativeLibraryProvider;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class ArgusCommonValidationAndSupportTest {

    @Test
    public void testNumericAndOverflowValidation() {
        assertDoesNotThrow(() -> ArgusCommonValidation.checkPositive(1, "val"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkPositive(0, "val"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkPositive(-5, "val"));

        assertDoesNotThrow(() -> ArgusCommonValidation.checkNonNegative(0, "val"));
        assertDoesNotThrow(() -> ArgusCommonValidation.checkNonNegative(42, "val"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkNonNegative(-1, "val"));

        assertEquals(0, ArgusCommonValidation.checkIntBounds(0, "idx"));
        assertEquals(Integer.MAX_VALUE, ArgusCommonValidation.checkIntBounds(Integer.MAX_VALUE, "idx"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkIntBounds(-1, "idx"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkIntBounds((long) Integer.MAX_VALUE + 1L, "idx"));

        assertEquals(40L, ArgusCommonValidation.multiplyExactBytes(10, 4, "tokens"));
        assertEquals(0L, ArgusCommonValidation.multiplyExactBytes(0, 4, "tokens"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.multiplyExactBytes(-1, 4, "tokens"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.multiplyExactBytes(10, 0, "tokens"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.multiplyExactBytes(Long.MAX_VALUE, 2, "tokens"));
    }

    @Test
    public void testDirectBufferAndUtf8Validation() {
        ByteBuffer heapBuf = ByteBuffer.allocate(16);
        ByteBuffer directBuf = ByteBuffer.allocateDirect(16);
        ByteBuffer readOnlyDirect = directBuf.asReadOnlyBuffer();

        assertThrows(NullPointerException.class, () -> ArgusCommonValidation.checkDirectBuffer(null, 4, "buf"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkDirectBuffer(heapBuf, 4, "buf"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkDirectBuffer(directBuf, 17, "buf"));
        assertDoesNotThrow(() -> ArgusCommonValidation.checkDirectBuffer(directBuf, 16, "buf"));
        assertDoesNotThrow(() -> ArgusCommonValidation.checkDirectBuffer(readOnlyDirect, 16, "buf"));

        assertDoesNotThrow(() -> ArgusCommonValidation.checkDirectWritableBuffer(directBuf, 16, "buf"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkDirectWritableBuffer(readOnlyDirect, 16, "buf"));

        byte[] utf8 = ArgusCommonValidation.toUtf8NoNull("hello 🚀", "str");
        assertArrayEquals("hello 🚀".getBytes(StandardCharsets.UTF_8), utf8);
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.toUtf8NoNull("bad\0str", "str"));
        assertDoesNotThrow(() -> ArgusCommonValidation.toUtf8("allowed\0nul", "str"));
    }

    @Test
    public void testBackendSupportMasksAndBuildInfoParsing() {
        assertEquals(1L, ArgusBackendSupport.expectedFeatureMask("cpu"));
        assertEquals(3L, ArgusBackendSupport.expectedFeatureMask("CUDA"));
        assertEquals(5L, ArgusBackendSupport.expectedFeatureMask("rocm"));
        assertEquals(5L, ArgusBackendSupport.expectedFeatureMask("hip"));
        assertEquals(9L, ArgusBackendSupport.expectedFeatureMask("vulkan"));
        assertEquals(17L, ArgusBackendSupport.expectedFeatureMask("metal"));
        assertThrows(IllegalArgumentException.class, () -> ArgusBackendSupport.expectedFeatureMask("unknown"));
        assertThrows(IllegalArgumentException.class, () -> ArgusBackendSupport.expectedFeatureMask(null));

        Map<String, String> parsed = ArgusBackendSupport.parseBuildInfo("version=2.1.1\ncpu_baseline=x86-64-v3\n", 9L);
        assertEquals("2.1.1", parsed.get("version"));
        assertEquals("x86-64-v3", parsed.get("cpu_baseline"));
        assertEquals("9", parsed.get("features"));

        assertThrows(IllegalStateException.class, () -> ArgusBackendSupport.parseBuildInfo("", 1L));
        assertThrows(IllegalStateException.class, () -> ArgusBackendSupport.parseBuildInfo("malformed_line", 1L));
        assertThrows(IllegalStateException.class, () -> ArgusBackendSupport.parseBuildInfo("k=1\nk=2\n", 1L));
    }

    @Test
    public void testArraySliceAndContextSlotValidation() {
        assertDoesNotThrow(() -> ArgusCommonValidation.checkArraySlice(10, 0, 10, "arr"));
        assertDoesNotThrow(() -> ArgusCommonValidation.checkArraySlice(10, 10, 0, "arr"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkArraySlice(10, -1, 5, "arr"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkArraySlice(10, 0, -1, "arr"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkArraySlice(10, 6, 5, "arr"));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.checkArraySlice(10, Integer.MAX_VALUE, 1, "arr"));

        assertDoesNotThrow(() -> ArgusCommonValidation.validateContextSlots(0, 0));
        assertDoesNotThrow(() -> ArgusCommonValidation.validateContextSlots(5, 4));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.validateContextSlots(4, 4));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.validateContextSlots(2, 4));

        assertEquals(0, ArgusCommonValidation.computeEffectiveSeqMax(0, 0, false));
        assertEquals(3, ArgusCommonValidation.computeEffectiveSeqMax(0, 2, false));
        assertEquals(6, ArgusCommonValidation.computeEffectiveSeqMax(0, 2, true));
        assertEquals(10, ArgusCommonValidation.computeEffectiveSeqMax(8, 2, false));
        assertThrows(IllegalArgumentException.class, () -> ArgusCommonValidation.computeEffectiveSeqMax(Integer.MAX_VALUE, 1, false));
        assertTrue(ArgusCommonValidation.defaultCpuThreads() >= 1);
    }

    @Test
    public void testSequenceSlotPoolLifecycleAndFailureRollback() {
        ArgusSequenceSlotPool pool = new ArgusSequenceSlotPool(3, 2); // Slot 0 base, slots 1 & 2 clone pool
        assertEquals(2, pool.getAvailableSlotCount());

        // Invalid source slot
        assertThrows(IllegalArgumentException.class, () -> pool.forkSlot(3, 3, (s, d) -> true));

        // Copy failure rolls back leased slot bit so count remains 2
        assertThrows(ArgusNativeException.class, () -> pool.forkSlot(0, 3, (s, d) -> false));
        assertEquals(2, pool.getAvailableSlotCount());

        // Lease both clone slots (1 and 2)
        int s1 = pool.forkSlot(0, 3, (s, d) -> true);
        int s2 = pool.forkSlot(0, 3, (s, d) -> true);
        assertEquals(1, s1);
        assertEquals(2, s2);
        assertEquals(0, pool.getAvailableSlotCount());

        // Pool exhausted
        assertThrows(IllegalStateException.class, () -> pool.forkSlot(0, 3, (s, d) -> true));

        // Cannot free root slot 0 or out-of-range slot 3
        assertThrows(IllegalArgumentException.class, () -> pool.freeSlot(0, slot -> true));
        assertThrows(IllegalArgumentException.class, () -> pool.freeSlot(3, slot -> true));

        // Failed clear retains leased state
        assertThrows(ArgusNativeException.class, () -> pool.freeSlot(s1, slot -> false));
        assertEquals(0, pool.getAvailableSlotCount());

        // Successful free restores availability; double-free is rejected
        pool.freeSlot(s1, slot -> true);
        assertEquals(1, pool.getAvailableSlotCount());
        assertThrows(IllegalStateException.class, () -> pool.freeSlot(s1, slot -> true));
    }

    @Test
    public void testTextGeneratorAndEarlyStopPredicate() {
        byte[][] vocabPieces = new byte[][]{
            "Hello".getBytes(StandardCharsets.UTF_8),
            " ".getBytes(StandardCharsets.UTF_8),
            "world".getBytes(StandardCharsets.UTF_8),
            "!".getBytes(StandardCharsets.UTF_8)
        };
        AtomicInteger decodeSteps = new AtomicInteger(0);
        AtomicInteger sampleSteps = new AtomicInteger(0);

        ArgusTextGenerator.ContextOps mockOps = new ArgusTextGenerator.ContextOps() {
            @Override public boolean clearCacheSlot(int seqId, int p0, int p1) { return true; }
            @Override public void resetSampler(int seqId) {}
            @Override public int[] tokenize(String text, boolean addBos) { return new int[]{10, 11}; }
            @Override public int decodeBatch(int[] tokens, int nTokens, int startPos, boolean requestLogits) {
                decodeSteps.incrementAndGet();
                return 0;
            }
            @Override public int primeSampler(int seqId, int[] tokens, int nTokens) { return nTokens; }
            @Override public int sampleToken(int seqId, ArgusSamplerConfig config) {
                return sampleSteps.getAndIncrement();
            }
            @Override public boolean vocabIsEog(int token) { return token >= vocabPieces.length; }
            @Override public byte[] tokenToPieceBytes(int token) { return vocabPieces[token]; }
            @Override public int nEmbd() { return 4; }
            @Override public int evalEmbeddings(int[] tokens, int nTokens, int seqId, float[] outEmbeddings) {
                for (int i = 0; i < 4; i++) outEmbeddings[i] = (i + 1) * 0.5f;
                return 4;
            }
        };

        // 1. Early stop after 2nd piece ("Hello ")
        List<String> pieces = new ArrayList<>();
        String partial = ArgusTextGenerator.generateUntil(
            mockOps,
            "prompt",
            10,
            ArgusSamplerConfig.greedy(),
            piece -> {
                pieces.add(piece);
                return pieces.size() < 2; // Stop on 2nd piece
            },
            null
        );
        assertEquals("Hello ", partial);
        assertEquals(List.of("Hello", " "), pieces);
        // 1 prefill decode + 1 step decode (after piece 1); step 2 halted before next decodeBatch!
        assertEquals(2, decodeSteps.get());

        // 2. Embeddings extraction
        float[] embd = ArgusTextGenerator.extractEmbeddings(mockOps, "embed me");
        assertArrayEquals(new float[]{0.5f, 1.0f, 1.5f, 2.0f}, embd, 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> ArgusTextGenerator.extractEmbeddings(mockOps, ""));
    }

    @Test
    public void testAbstractNativeLibraryProvider() {
        AbstractNativeLibraryProvider provider = new AbstractNativeLibraryProvider("linux", "amd64", "cuda", 100, "libargus.so") {};
        assertEquals("linux", provider.getOs());
        assertEquals("amd64", provider.getArch());
        assertEquals("cuda", provider.getBackend());
        assertEquals(100, provider.getPriority());
        assertEquals("libargus.so", provider.getLibraryName());
    }
}
