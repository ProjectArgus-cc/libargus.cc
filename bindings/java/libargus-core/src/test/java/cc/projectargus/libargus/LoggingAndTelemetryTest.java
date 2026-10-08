package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import cc.projectargus.libargus.internal.ArgusBindings;
import static org.junit.jupiter.api.Assertions.*;

class LoggingAndTelemetryTest {

    private Path modelPath() {
        String root = System.getProperty("argus.root", ".");
        return Path.of(root, "tests/data/tiny.gguf");
    }

    @Test
    void testLogLevelGetSetAndQuiet() {
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        assertNotNull(initial);

        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            assertEquals(ArgusLogLevel.DEBUG, ArgusBackend.getLogLevel());

            ArgusBackend.setLogLevel(ArgusLogLevel.NONE);
            assertEquals(ArgusLogLevel.NONE, ArgusBackend.getLogLevel());

            ArgusBackend.setQuiet(false);
            assertEquals(ArgusLogLevel.WARN, ArgusBackend.getLogLevel());

            ArgusBackend.setQuiet(true);
            assertEquals(ArgusLogLevel.NONE, ArgusBackend.getLogLevel());

            assertThrows(IllegalArgumentException.class, () -> ArgusBackend.setLogLevel(null));
        } finally {
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testLogCallbackUpcall() {
        record LogEntry(ArgusLogLevel level, String message) {}
        List<LogEntry> logs = new CopyOnWriteArrayList<>();

        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            ArgusBackend.setLogCallback((level, message) -> logs.add(new LogEntry(level, message)));

            // Trigger an operation that generates a native error message
            try (Arena arena = Arena.ofConfined()) {
                assertThrows(Throwable.class, () -> {
                    ArgusModel.load(arena, Path.of("non_existent_model_for_test.gguf"), 0, false);
                });
            }

            assertFalse(logs.isEmpty(), "Log callback should have received messages during failed model load");
            boolean foundMessage = logs.stream().anyMatch(e ->
                e.message().contains("non_existent_model") ||
                e.message().toLowerCase().contains("failed") ||
                e.message().toLowerCase().contains("error")
            );
            assertTrue(foundMessage, "Captured log should contain error description");

            // Clear callback and verify cleanup
            ArgusBackend.setLogCallback(null);
        } finally {
            ArgusBackend.setLogCallback(null);
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testPerformanceTelemetryDuringGeneration() throws Exception {
        ArgusBackend.init();
        try (Arena setup = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(setup, modelPath(), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128))) {

            // Initial telemetry query
            ArgusPerfTimings initialTimings = ctx.getPerfTimings();
            assertNotNull(initialTimings);
            assertTrue(initialTimings.startTimeMs() > 0.0);

            // Evaluate a prompt batch
            MemorySegment tokens = setup.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5, 6);
            assertEquals(0, ctx.decodeBatch(tokens, 4, 0, 0, false));

            ArgusPerfTimings evaluatedTimings = ctx.getPerfTimings();
            assertNotNull(evaluatedTimings);
            assertTrue(evaluatedTimings.nPromptEval() >= 4, "Prompt tokens count should be >= 4");
            assertTrue(evaluatedTimings.promptEvalTimeMs() >= 0.0);
            assertTrue(evaluatedTimings.promptTokensPerSecond() >= 0.0);

            // Reset performance telemetry
            ctx.resetPerfTimings();
            ArgusPerfTimings resetTimings = ctx.getPerfTimings();
            assertNotNull(resetTimings);
            assertTrue(resetTimings.nPromptEval() < evaluatedTimings.nPromptEval(),
                "Prompt tokens count should be reset after clear");
            assertEquals(0.0, resetTimings.promptEvalTimeMs(), 0.001);
        } finally {
            ArgusBackend.free();
        }
    }

    @Test
    void testLogLevelErrorSuppressesWarnings() {
        record LogEntry(ArgusLogLevel level, String message) {}
        List<LogEntry> logs = new CopyOnWriteArrayList<>();

        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.ERROR);
            ArgusBackend.setLogCallback((level, message) -> logs.add(new LogEntry(level, message)));

            ArgusBackend.init();
            try (Arena arena = Arena.ofConfined();
                 ArgusModel model = ArgusModel.load(arena, modelPath(), 0, false)) {
                assertNotNull(model);
            } finally {
                ArgusBackend.free();
            }

            // tiny.gguf triggers a WARN in SPM vocab loading ("SPM vocabulary, but newline token not found")
            // With log level ERROR, zero warnings or info messages should be delivered to the callback!
            boolean hasWarnOrInfo = logs.stream().anyMatch(e ->
                e.level() == ArgusLogLevel.WARN || e.level() == ArgusLogLevel.INFO || e.level() == ArgusLogLevel.DEBUG
            );
            assertFalse(hasWarnOrInfo, "Log level ERROR must suppress SPM vocabulary warning from tiny.gguf");
        } finally {
            ArgusBackend.setLogCallback(null);
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testLogCallbackConcurrentReplacementAndStress() throws Exception {
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            java.util.concurrent.atomic.AtomicBoolean running = new java.util.concurrent.atomic.AtomicBoolean(true);
            java.util.concurrent.atomic.AtomicInteger logCountA = new java.util.concurrent.atomic.AtomicInteger(0);
            java.util.concurrent.atomic.AtomicInteger logCountB = new java.util.concurrent.atomic.AtomicInteger(0);
            java.util.concurrent.atomic.AtomicInteger logCountC = new java.util.concurrent.atomic.AtomicInteger(0);
            java.util.concurrent.atomic.AtomicReference<Throwable> workerError = new java.util.concurrent.atomic.AtomicReference<>();

            ArgusLogCallback cbA = (level, msg) -> logCountA.incrementAndGet();
            ArgusLogCallback cbB = (level, msg) -> logCountB.incrementAndGet();
            ArgusLogCallback cbC = (level, msg) -> logCountC.incrementAndGet();

            Thread loggerThread = new Thread(() -> {
                try (Arena local = Arena.ofConfined()) {
                    MemorySegment textSeg = local.allocateFrom("Worker stress log line\n");
                    while (running.get()) {
                        ArgusBindings.argus_test_emit_log.invokeExact(1, textSeg);
                    }
                } catch (Throwable t) {
                    workerError.set(t);
                }
            });

            loggerThread.start();

            // Rapidly swap callbacks across thousands of iterations
            for (int i = 0; i < 2000; i++) {
                ArgusBackend.setLogCallback(cbA);
                ArgusBackend.setLogCallback(cbB);
                ArgusBackend.setLogCallback(null);
                ArgusBackend.setLogCallback(cbC);
                ArgusBackend.setLogCallback(null);
            }

            running.set(false);
            loggerThread.join(5000);
            assertFalse(loggerThread.isAlive(), "Logger thread should terminate cleanly");
            assertNull(workerError.get(), "Worker thread encountered an exception");

            // Final quiet check: ensure arenas drain safely once in_flight is 0
            ArgusBackend.setLogCallback(null);
            ArgusBackend.drainRetiredCallbackArenas();
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(), "All retired arenas should drain after quiescence");
        } finally {
            ArgusBackend.setLogCallback(null);
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testReentrantCallbackReplacement() throws Throwable {
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            java.util.concurrent.atomic.AtomicInteger countA = new java.util.concurrent.atomic.AtomicInteger(0);
            java.util.concurrent.atomic.AtomicInteger countB = new java.util.concurrent.atomic.AtomicInteger(0);

            ArgusLogCallback cbB = (level, msg) -> countB.incrementAndGet();

            // cbA replaces the active callback with cbB from WITHIN its own invocation
            ArgusLogCallback cbA = (level, msg) -> {
                countA.incrementAndGet();
                ArgusBackend.setLogCallback(cbB);
            };

            ArgusBackend.setLogCallback(cbA);

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment msg1 = arena.allocateFrom("First message triggers reentrant swap\n");
                ArgusBindings.argus_test_emit_log.invokeExact(1, msg1);

                MemorySegment msg2 = arena.allocateFrom("Second message goes to cbB\n");
                ArgusBindings.argus_test_emit_log.invokeExact(1, msg2);
            }

            assertEquals(1, countA.get(), "cbA should execute once");
            assertTrue(countB.get() >= 1, "cbB should receive the second message");

            ArgusBackend.setLogCallback(null);
            ArgusBackend.drainRetiredCallbackArenas();
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(), "Retired arenas should drain cleanly");
        } finally {
            ArgusBackend.setLogCallback(null);
            ArgusBackend.setLogLevel(initial);
        }
    }
}

