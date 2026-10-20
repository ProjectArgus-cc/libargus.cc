package cc.projectargus.libargus.internal;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;

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
}
