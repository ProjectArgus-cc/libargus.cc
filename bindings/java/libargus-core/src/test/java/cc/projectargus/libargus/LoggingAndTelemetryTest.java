package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
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
}
