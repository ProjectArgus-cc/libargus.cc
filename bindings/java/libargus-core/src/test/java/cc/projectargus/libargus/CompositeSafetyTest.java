package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusNativeResource;
import cc.projectargus.libargus.internal.LifecycleProbe;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.lang.foreign.*;
import java.lang.invoke.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named="argus.testing", matches="true")
class CompositeSafetyTest {
    private Path fixture(String name) { return Path.of(System.getProperty("argus.root"), "tests/data", name); }

    @Test void publicMediaEvaluationLeasesEveryDependency() throws Throwable {
        ArgusBackend.init();
        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, fixture("tiny.gguf"), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128));
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(arena, model, fixture("tiny-mmproj.gguf"), 2, false);
             ArgusBitmap bitmap = ArgusBitmap.fromRgb(16, 16, arena.allocate(16*16*3).fill((byte)127));
             ArgusInputChunks chunks = mctx.tokenize(arena, "<__media__></s>", false, List.of(bitmap))) {
            heldCall(3, () -> assertEquals(5, ctx.evalMultimodalChunks(mctx, chunks, 0, 0, 8, true)), ctx, mctx, chunks);
            assertTrue(ctx.isClosed()); assertTrue(mctx.isClosed()); assertTrue(chunks.isClosed());
        } finally { ArgusBackend.free(); }
    }

    @Test void publicSpeechOperationLeasesVocoderThroughErrorPath() throws Throwable {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined(); Arena buffers = Arena.ofShared();
             ArgusModel model = ArgusModel.load(setup, fixture("tiny.gguf"), 0, false);
             ArgusModel vocoder = ArgusModel.load(setup, fixture("tiny.gguf"), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128))) {
            MemorySegment text = buffers.allocateFrom("hello");
            MemorySegment pcm = buffers.allocate(128 * 4, 4);
            // A text model is deliberately incompatible with speech. Its failed
            // public call must still keep the supplied vocoder alive throughout.
            heldCall(9, () -> assertThrows(ArgusNativeException.class,
                    () -> ctx.synthesizeSpeech(vocoder, text, 0, pcm, 128)), vocoder);
            assertTrue(vocoder.isClosed());
        } finally { ArgusBackend.free(); }
    }

    @Test void publicVideoReadExcludesCloseWithoutFfmpeg() throws Throwable {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(setup, fixture("tiny.gguf"), 0, false);
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(setup, model, fixture("tiny-mmproj.gguf"), 2, false)) {
            MethodHandle factory = Linker.nativeLinker().downcallHandle(SymbolLookup.loaderLookup()
                .find("argus_test_video").orElseThrow(), FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            MemorySegment pointer;
            try (var lease = mctx.lease()) { pointer = (MemorySegment) factory.invokeExact(lease.handle(), 2); }
            var constructor = ArgusVideo.class.getDeclaredConstructor(MemorySegment.class, ArgusMultimodalContext.class);
            constructor.setAccessible(true);
            try (ArgusVideo video = constructor.newInstance(pointer, mctx)) {
                mctx.close(); // The native video retains the projector independently.
                heldCall(5, () -> assertEquals("0", video.readNext().text()), video);
                assertTrue(video.isClosed());
            }
        } finally { ArgusBackend.free(); }
    }

    private void heldCall(int event, Runnable operation, ArgusNativeResource... dependencies) throws Throwable {
        try (Arena hooks = Arena.ofShared()) {
            Gate gate = new Gate(event);
            MethodHandle setter = Linker.nativeLinker().downcallHandle(SymbolLookup.loaderLookup()
                .find("argus_test_set_observer").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            MethodHandle callback = MethodHandles.lookup().findVirtual(Gate.class, "observe",
                MethodType.methodType(void.class, int.class, MemorySegment.class)).bindTo(gate);
            MemorySegment stub = Linker.nativeLinker().upcallStub(callback,
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS), hooks);
            setter.invokeExact(stub);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread worker = new Thread(() -> { try { operation.run(); } catch (Throwable t) { failure.set(t); } });
            List<Thread> closers = new ArrayList<>();
            try {
                worker.start(); assertTrue(gate.entered.await(5, TimeUnit.SECONDS), "Native operation did not reach barrier");
                for (ArgusNativeResource resource : dependencies) {
                    Thread closer = new Thread(() -> { try { resource.close(); } catch (Throwable t) { failure.set(t); } });
                    closers.add(closer); closer.start(); LifecycleProbe.awaitQueued(resource, closer);
                    assertFalse(resource.isClosed());
                }
            } finally {
                gate.exit.countDown(); worker.join(10000);
                for (Thread closer : closers) closer.join(10000);
                setter.invokeExact(MemorySegment.NULL);
            }
            assertFalse(worker.isAlive());
            for (Thread closer : closers) assertFalse(closer.isAlive());
            assertNull(gate.failure.get()); assertNull(failure.get());
        }
    }

    private static final class Gate {
        final int target;
        final CountDownLatch entered = new CountDownLatch(1), exit = new CountDownLatch(1);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        Gate(int target) { this.target = target; }
        public void observe(int event, MemorySegment ignored) {
            if (event != target) return;
            entered.countDown();
            try { if (!exit.await(10, TimeUnit.SECONDS)) failure.set(new AssertionError("Native barrier timed out")); }
            catch (Throwable t) { failure.set(t); }
        }
    }
}
