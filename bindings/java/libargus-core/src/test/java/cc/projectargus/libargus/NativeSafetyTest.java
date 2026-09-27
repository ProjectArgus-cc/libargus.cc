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

    @Test
    void testSequenceForkingAndSlotLeasingLifecycle() throws Exception {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(setup, modelPath(), 0, false)) {

            // Topology queries on standard transformer architecture
            assertFalse(model.isRecurrent());
            assertFalse(model.isHybrid());
            assertFalse(model.isDiffusion());

            ArgusContextConfig config = new ArgusContextConfig.Builder(128)
                .seqMax(1)
                .cloneSlots(2)
                .build();
            assertEquals(3, config.seqMax());

            try (ArgusContext ctx = ArgusContext.init(model, config)) {
                assertEquals(3, ctx.getSeqMax());
                assertTrue(ctx.canShift());
                assertFalse(ctx.isRecurrent());
                assertFalse(ctx.isHybrid());
                assertFalse(ctx.isDiffusion());
                assertEquals(2, ctx.getAvailableSlotCount());

                // Prefill prompt on root slot 0
                MemorySegment prompt = setup.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5);
                assertEquals(0, ctx.decodeBatch(prompt, 3, 0, 0, false));
                assertEquals(2, ctx.getSeqPosMax(0));
                assertEquals(0, ctx.getSeqPosMin(0));

                // Fork slot 0 into dedicated clone worker 1
                int worker1 = ctx.forkSlot(0);
                assertEquals(1, worker1);
                assertEquals(1, ctx.getAvailableSlotCount());
                assertEquals(2, ctx.getSeqPosMax(worker1));
                assertEquals(0, ctx.getSeqPosMin(worker1));

                // Decode token on worker1; verify position advances independently without mutating slot 0
                MemorySegment tok1 = setup.allocateFrom(ValueLayout.JAVA_INT, 6);
                assertEquals(0, ctx.decodeBatch(tok1, 1, 3, worker1, false));
                assertEquals(3, ctx.getSeqPosMax(worker1));
                assertEquals(2, ctx.getSeqPosMax(0));

                // Fork slot 0 into dedicated clone worker 2
                int worker2 = ctx.forkSlot(0);
                assertEquals(2, worker2);
                assertEquals(0, ctx.getAvailableSlotCount());
                assertEquals(2, ctx.getSeqPosMax(worker2));

                // Pool exhaustion throws IllegalStateException
                assertThrows(IllegalStateException.class, () -> ctx.forkSlot(0));

                // Free worker1 and verify slot is cleared
                ctx.freeSlot(worker1);
                assertEquals(1, ctx.getAvailableSlotCount());
                assertEquals(-1, ctx.getSeqPosMax(worker1));

                // Re-leasing slot succeeds and reuses worker1
                int worker3 = ctx.forkSlot(worker2);
                assertEquals(worker1, worker3);
                assertEquals(0, ctx.getAvailableSlotCount());
                assertEquals(2, ctx.getSeqPosMax(worker3));

                // Bounds and invariant enforcement
                assertThrows(IllegalArgumentException.class, () -> ctx.freeSlot(0));
                assertThrows(IllegalArgumentException.class, () -> ctx.freeSlot(-1));
                assertThrows(IllegalArgumentException.class, () -> ctx.freeSlot(99));
                assertThrows(IllegalArgumentException.class, () -> ctx.forkSlot(99));
                assertThrows(IllegalArgumentException.class, () -> ctx.forkSlot(-1));

                // Direct slot copy bounds check returns false rather than native abort
                assertFalse(ctx.copySequenceSlot(0, 99, 0, -1));
                assertFalse(ctx.clearCacheSlot(99, 0, -1));
                assertTrue(ctx.clearCacheSlot(worker2, 0, -1));

                ctx.freeSlot(worker2);
                ctx.freeSlot(worker3);
                assertEquals(2, ctx.getAvailableSlotCount());
            }
        } finally {
            ArgusBackend.free();
        }
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
