package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.LifecycleProbe;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;

class NativeSafetyTest {
    private Path modelPath() { return Path.of(System.getProperty("argus.root"), "tests/data/tiny.gguf"); }

    @Test void directBuffersRejectInvalidAccessBeforeDecode() throws Exception {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(setup, modelPath(), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128))) {
            MemorySegment good = setup.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5);
            assertEquals(0, ctx.decodeBatch(good.asReadOnly(), 3, 0, 0, false));
            int position = ctx.getSeqPosMax(0);
            assertThrows(IllegalArgumentException.class, () -> ctx.decodeBatch(good.asSlice(1), 1, 3, 0, false));
            assertThrows(IllegalArgumentException.class, () -> ctx.decodeBatch(good.asSlice(0, 2), 1, 3, 0, false));
            assertThrows(IllegalArgumentException.class, () -> ctx.decodeBatch(MemorySegment.ofArray(new int[]{1}), 1, 3, 0, false));
            MemorySegment expired;
            try (Arena scope = Arena.ofConfined()) { expired = scope.allocate(ValueLayout.JAVA_INT); }
            MemorySegment closed = expired;
            assertThrows(IllegalStateException.class, () -> ctx.decodeBatch(closed, 1, 3, 0, false));
            var error = new AtomicReference<Throwable>();
            Thread wrongThread = new Thread(() -> {
                try { ctx.decodeBatch(good, 3, 3, 0, false); }
                catch (Throwable t) { error.set(t); }
            });
            wrongThread.start(); wrongThread.join(5000);
            assertFalse(wrongThread.isAlive()); assertInstanceOf(WrongThreadException.class, error.get());
            assertEquals(position, ctx.getSeqPosMax(0));
        } finally { ArgusBackend.free(); }
    }

    @Test
    @EnabledIfSystemProperty(named="argus.testing", matches="true")
    void nativeDecodeKeepsBufferAndPublicAbortLeaseAlive() throws Throwable {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined(); Arena bufferArena = Arena.ofShared(); Arena hooks = Arena.ofShared();
             ArgusModel model = ArgusModel.load(setup, modelPath(), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128));
             ArgusAbortFlag abort = new ArgusAbortFlag()) {
            MemorySegment tokens = bufferArena.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5);
            var setter = Linker.nativeLinker().downcallHandle(SymbolLookup.loaderLookup()
                .find("argus_test_set_observer").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            Gate gate = new Gate();
            var callback = MethodHandles.lookup().findVirtual(Gate.class, "observe",
                MethodType.methodType(void.class, int.class, MemorySegment.class)).bindTo(gate);
            var stub = Linker.nativeLinker().upcallStub(callback,
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS), hooks);
            setter.invokeExact(stub);
            var failure = new AtomicReference<Throwable>();
            Thread worker = new Thread(() -> {
                try { assertEquals(0, ctx.decodeBatch(tokens, 3, 0, 0, false, abort)); }
                catch (Throwable t) { failure.set(t); }
            });
            Thread closer = new Thread(() -> {
                try { abort.close(); } catch (Throwable t) { failure.set(t); }
            });
            try {
                worker.start(); assertTrue(gate.entered.await(5, TimeUnit.SECONDS));
                closer.start(); LifecycleProbe.awaitQueued(abort, closer);
                assertFalse(abort.isClosed());
                assertThrows(IllegalStateException.class, bufferArena::close);
            } finally {
                gate.exit.countDown();
                worker.join(10000); closer.join(10000);
                setter.invokeExact(MemorySegment.NULL);
            }
            assertFalse(worker.isAlive()); assertFalse(closer.isAlive());
            assertNull(failure.get()); assertNull(gate.failure.get());
            assertTrue(abort.isClosed()); assertEquals(2, ctx.getSeqPosMax(0));
            // Scope closure is now permitted; try-with-resources closes it once below.
        } finally { ArgusBackend.free(); }
    }

    private static final class Gate {
        final CountDownLatch entered = new CountDownLatch(1), exit = new CountDownLatch(1);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        public void observe(int event, MemorySegment resource) {
            if (event != 4) return;
            entered.countDown();
            try {
                if (!exit.await(10, TimeUnit.SECONDS)) failure.set(new AssertionError("Native barrier timed out"));
            } catch (Throwable t) { failure.set(t); }
        }
    }
}
