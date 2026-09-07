package cc.projectargus.libargus.internal;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ArgusInternalResourceTest {

    @Test
    public void testCriticalAllowlistEnforcement() {
        assertEquals(4, ArgusBindings.CRITICAL_ALLOWLIST.size());
        assertTrue(ArgusBindings.CRITICAL_ALLOWLIST.contains("argus_build_features"));
        assertTrue(ArgusBindings.CRITICAL_ALLOWLIST.contains("argus_abort_flag_is_requested"));
        assertTrue(ArgusBindings.CRITICAL_ALLOWLIST.contains("argus_last_error_code"));
        assertTrue(ArgusBindings.CRITICAL_ALLOWLIST.contains("argus_clear_error"));

        // Verify that critical downcall on a non-allowlisted symbol fails fast via SecurityException
        assertThrows(SecurityException.class, () -> {
            ArgusBindings.bindCritical("argus_backend_get_count", FunctionDescriptor.of(ValueLayout.JAVA_INT));
        });
    }

    @Test
    public void testWithHandleInternalExecution() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dummyPtr = arena.allocate(16);
            ArgusNativeResource res = new ArgusNativeResource(dummyPtr) {
                @Override
                protected String resourceName() { return "TestResource"; }
                @Override
                protected void releaseNative(MemorySegment oldHandle) {}
            };
            long addr = res.withHandle(MemorySegment::address);
            assertEquals(dummyPtr.address(), addr);

            res.close();
            assertThrows(IllegalStateException.class, () -> res.withHandle(h -> h));
        }
    }
}
