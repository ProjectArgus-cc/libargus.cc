package cc.projectargus.libargus.internal;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Test-only FFM downcall bindings for instrumented test native libraries (argus_test).
 * Dynamically resolves symbols from the process loader lookup so production builds
 * remain unpolluted by test ABI symbols.
 */
public final class ArgusTestBindings {

    private static final MethodHandle MH_ARGUS_TEST_EMIT_LOG;
    private static final MethodHandle MH_ARGUS_TEST_SET_FORCE_ROLLBACK_FAIL;

    static {
        MethodHandle emitLog = null;
        MethodHandle rollbackFail = null;
        try {
            SymbolLookup lookup = SymbolLookup.loaderLookup();
            var optEmit = lookup.find("argus_test_emit_log");
            if (optEmit.isPresent()) {
                emitLog = Linker.nativeLinker().downcallHandle(
                    optEmit.get(),
                    FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
                );
            }
            var optRollback = lookup.find("argus_test_set_force_rollback_fail");
            if (optRollback.isPresent()) {
                rollbackFail = Linker.nativeLinker().downcallHandle(
                    optRollback.get(),
                    FunctionDescriptor.ofVoid(ValueLayout.JAVA_BOOLEAN)
                );
            }
        } catch (Throwable ignored) {}
        MH_ARGUS_TEST_EMIT_LOG = emitLog;
        MH_ARGUS_TEST_SET_FORCE_ROLLBACK_FAIL = rollbackFail;
    }

    private ArgusTestBindings() {}

    /**
     * @return true if test-only native symbols are resolved in the current native runtime
     */
    public static boolean hasTestHooks() {
        return MH_ARGUS_TEST_EMIT_LOG != null;
    }

    /**
     * Emits a native log message directly into the logging dispatch pipeline.
     */
    public static void emitLog(int level, MemorySegment text) throws Throwable {
        if (MH_ARGUS_TEST_EMIT_LOG == null) {
            throw new UnsupportedOperationException("argus_test_emit_log is not exported by the loaded native library");
        }
        MH_ARGUS_TEST_EMIT_LOG.invokeExact(level, text);
    }

    /**
     * Toggles deterministic KV cache rollback failure injection.
     */
    public static void setForceRollbackFail(boolean fail) throws Throwable {
        if (MH_ARGUS_TEST_SET_FORCE_ROLLBACK_FAIL == null) {
            throw new UnsupportedOperationException("argus_test_set_force_rollback_fail is not exported by the loaded native library");
        }
        MH_ARGUS_TEST_SET_FORCE_ROLLBACK_FAIL.invokeExact(fail);
    }
}
