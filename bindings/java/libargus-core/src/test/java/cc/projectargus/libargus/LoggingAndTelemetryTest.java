package cc.projectargus.libargus;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import cc.projectargus.libargus.internal.ArgusBindings;
import cc.projectargus.libargus.internal.ArgusTestBindings;
import org.junit.jupiter.api.Assumptions;
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
            assertTrue(initialTimings.nReallocsSame() >= 0);
            assertTrue(initialTimings.nReallocsDiff() >= 0);

            // Evaluate a prompt batch
            MemorySegment tokens = setup.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5, 6);
            assertEquals(0, ctx.decodeBatch(tokens, 4, 0, 0, false));

            ArgusPerfTimings evaluatedTimings = ctx.getPerfTimings();
            assertNotNull(evaluatedTimings);
            assertTrue(evaluatedTimings.nPromptEval() >= 4, "Prompt tokens count should be >= 4");
            assertTrue(evaluatedTimings.promptEvalTimeMs() >= 0.0);
            assertTrue(evaluatedTimings.promptTokensPerSecond() >= 0.0);
            assertTrue(evaluatedTimings.nReallocsSame() >= 0);
            assertTrue(evaluatedTimings.nReallocsDiff() >= 0);

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
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required for stress test");
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
                        ArgusTestBindings.emitLog(1, textSeg);
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
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required for reentrant replacement");
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
                ArgusTestBindings.emitLog(1, msg1);

                MemorySegment msg2 = arena.allocateFrom("Second message goes to cbB\n");
                ArgusTestBindings.emitLog(1, msg2);
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

    @Test
    void testSlowCallbackExceedingShutdownTimeout() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required for slow-callback test");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            java.util.concurrent.CountDownLatch callbackEntered = new java.util.concurrent.CountDownLatch(1);
            java.util.concurrent.CountDownLatch releaseCallback = new java.util.concurrent.CountDownLatch(1);
            java.util.concurrent.atomic.AtomicBoolean callbackFinished = new java.util.concurrent.atomic.AtomicBoolean(false);
            java.util.concurrent.atomic.AtomicReference<Throwable> callbackError = new java.util.concurrent.atomic.AtomicReference<>();

            ArgusBackend.setLogCallback((level, msg) -> {
                callbackEntered.countDown();
                try {
                    // Block callback until latch is released
                    releaseCallback.await(5000, java.util.concurrent.TimeUnit.MILLISECONDS);
                    callbackFinished.set(true);
                } catch (Throwable t) {
                    callbackError.set(t);
                }
            });

            // Worker thread emits log triggering the callback
            Thread worker = new Thread(() -> {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment text = arena.allocateFrom("Slow callback trigger line\n");
                    ArgusTestBindings.emitLog(1, text);
                } catch (Throwable t) {
                    callbackError.set(t);
                }
            });
            worker.start();

            // Wait for callback to be actively executing
            assertTrue(callbackEntered.await(5000, java.util.concurrent.TimeUnit.MILLISECONDS), "Callback should have entered");

            // Call ArgusBackend.free() from main thread while callback is still running.
            // free() must wait for timeout (~1000ms).
            long startFree = System.currentTimeMillis();
            ArgusBackend.free();
            long freeElapsed = System.currentTimeMillis() - startFree;

            assertTrue(freeElapsed >= 900, "free() should wait up to deadline for in-flight callback");

            // CRITICAL INVARIANT: The callback is STILL running, so in-flight count > 0.
            // Its retired arena MUST NOT have been closed!
            assertTrue(ArgusBackend.getRetiredCallbackArenaCount() > 0, "Retired arena must NOT be closed while callback is still executing");
            assertFalse(callbackFinished.get(), "Callback should still be blocked");

            // Release the callback so it can safely complete
            releaseCallback.countDown();
            worker.join(5000);
            assertFalse(worker.isAlive(), "Worker thread should complete cleanly");
            assertTrue(callbackFinished.get(), "Callback should finish cleanly without JVM crash");
            assertNull(callbackError.get(), "Callback encountered error: " + callbackError.get());

            // Do NOT call drainRetiredCallbackArenas() manually!
            // Wait for automatic post-quiescence virtual thread reaper to reclaim the retired arenas
            long reaperDeadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < reaperDeadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "Retired arenas must be reclaimed automatically by post-quiescence reaper without manual drain");
        } finally {
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testReentrantFreeFromCallback() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required for reentrant free test");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            java.util.concurrent.atomic.AtomicBoolean callbackExecuted = new java.util.concurrent.atomic.AtomicBoolean(false);
            java.util.concurrent.atomic.AtomicReference<Throwable> callbackError = new java.util.concurrent.atomic.AtomicReference<>();

            ArgusBackend.setLogCallback((level, msg) -> {
                try {
                    callbackExecuted.set(true);
                    // Direct reentrant invocation of free() from within the active log callback
                    ArgusBackend.free();
                } catch (Throwable t) {
                    callbackError.set(t);
                }
            });

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment text = arena.allocateFrom("Reentrant free trigger\n");
                ArgusTestBindings.emitLog(1, text);
            }

            assertTrue(callbackExecuted.get(), "Log callback should have executed");
            assertNull(callbackError.get(), "Reentrant free should not throw: " + callbackError.get());

            // Do NOT call drainRetiredCallbackArenas() manually!
            long reaperDeadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < reaperDeadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "Arenas should drain automatically via reaper after reentrant free");
        } finally {
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testRepeatedInitAndCallbackRetirementStress() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required for stress test");
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);
            // Run 100+ rapid cycles of init -> setLogCallback -> emit log -> free
            for (int cycle = 0; cycle < 120; cycle++) {
                ArgusBackend.init();
                java.util.concurrent.atomic.AtomicInteger count = new java.util.concurrent.atomic.AtomicInteger(0);
                ArgusBackend.setLogCallback((lvl, msg) -> count.incrementAndGet());
                try (Arena a = Arena.ofConfined()) {
                    ArgusTestBindings.emitLog(1, a.allocateFrom("Cycle " + cycle + "\n"));
                }
                assertTrue(count.get() >= 1, "Log should be received in cycle " + cycle);
                ArgusBackend.free();
            }

            // Assert retired queue remains strictly bounded and drops to 0 automatically
            long deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "Retired arena count must return to 0 after repeated lifecycle stress without accumulation");
        } finally {
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testStaleZeroObservationRace() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);

            // Step 1: Register initial callback X and execute it on a worker thread
            CountDownLatch xEntered = new CountDownLatch(1);
            CountDownLatch xRelease = new CountDownLatch(1);
            CountDownLatch xDone = new CountDownLatch(1);

            ArgusBackend.setLogCallback((lvl, msg) -> {
                if (msg.contains("Message invoking X")) {
                    xEntered.countDown();
                    try {
                        xRelease.await();
                    } catch (InterruptedException ignored) {}
                    xDone.countDown();
                }
            });

            Thread workerX = Thread.ofPlatform().start(() -> {
                try (Arena a = Arena.ofConfined()) {
                    ArgusTestBindings.emitLog(1, a.allocateFrom("Message invoking X\n"));
                } catch (Throwable ignored) {}
            });
            assertTrue(xEntered.await(5, TimeUnit.SECONDS), "Callback X should enter");

            // Step 2: Register callback A while X is still running.
            // This retires callback X into retiredCallbacks (Gen 1) while in_flight == 1.
            CountDownLatch aEntered = new CountDownLatch(1);
            CountDownLatch aRelease = new CountDownLatch(1);
            AtomicBoolean aFinished = new AtomicBoolean(false);
            AtomicReference<Throwable> aError = new AtomicReference<>();

            ArgusBackend.setLogCallback((lvl, msg) -> {
                if (msg.contains("Message invoking A")) {
                    try {
                        aEntered.countDown();
                        if (!aRelease.await(5, TimeUnit.SECONDS)) {
                            throw new RuntimeException("Timed out waiting for callback A release");
                        }
                        aFinished.set(true);
                    } catch (Throwable t) {
                        aError.set(t);
                    }
                }
            });

            // At this point:
            // Callback X is retired (Gen 1) and reaper is running.
            // Callback A is active.
            assertEquals(1, ArgusBackend.getRetiredCallbackArenaCount(), "Generation X should be in retired queue");

            // Step 3: Setup deterministic hook to pause the reaper between quiescence check and drainage
            CountDownLatch hookTriggered = new CountDownLatch(1);
            CountDownLatch workerStarted = new CountDownLatch(1);

            ArgusBackend.testReaperPostQuiescenceHook = (eligibleGen) -> {
                try {
                    if (hookTriggered.getCount() > 0) {
                        hookTriggered.countDown();
                        // 1. Launch native log invocation in worker thread to execute Callback A
                        Thread workerA = Thread.ofPlatform().start(() -> {
                            workerStarted.countDown();
                            try (Arena a = Arena.ofConfined()) {
                                ArgusTestBindings.emitLog(1, a.allocateFrom("Message invoking A\n"));
                            } catch (Throwable t) {
                                aError.set(t);
                            }
                        });

                        // 2. Wait until Callback A is actively executing inside the upcall
                        assertTrue(aEntered.await(5, TimeUnit.SECONDS), "Callback A should enter");

                        // 3. Concurrently replace Callback A with Callback B!
                        // This retires Callback A into retiredCallbacks as Gen 2 while it is STILL executing!
                        ArgusBackend.setLogCallback((lvl, msg) -> {});

                        // Callback A is now in retiredCallbacks (Gen 2), while reaper has snapshot eligibleGen == 1!
                    }
                } catch (Throwable t) {
                    throw new RuntimeException(t);
                }
            };

            // Step 4: Release Callback X so native in_flight drops to 0, causing reaper to observe quiescence
            xRelease.countDown();
            workerX.join(5000);
            assertTrue(xDone.await(5, TimeUnit.SECONDS), "Callback X should complete");

            // Step 5: Wait for the hook to trigger and inject the race
            assertTrue(hookTriggered.await(5, TimeUnit.SECONDS), "Reaper hook should be invoked on quiescence");

            // Step 6: Reaper proceeds to reclaimGenerationsUpTo(1).
            // CRITICAL INVARIANT:
            // Gen 1 (Callback X) is reclaimed.
            // Gen 2 (Callback A) MUST NOT be reclaimed because it was retired AFTER the quiescence point and is still running!
            Thread.sleep(100);

            // Assert: Arena A is still alive and in retiredCallbacks!
            assertEquals(1, ArgusBackend.getRetiredCallbackArenaCount(),
                "Arena A must NOT be destroyed by the reaper pass that captured quiescence before A was retired!");
            assertFalse(aFinished.get(), "Callback A should still be actively blocked inside native call");

            // Step 7: Release Callback A so it completes cleanly
            aRelease.countDown();

            // Step 8: Wait for quiescence and subsequent reaper pass to reclaim Gen 2 (Callback A)
            long deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }

            assertTrue(aFinished.get(), "Callback A should finish cleanly without arena closure exception");
            assertNull(aError.get(), "Callback A encountered error: " + aError.get());
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "All generations must eventually be reclaimed once native execution quiesces");
        } finally {
            ArgusBackend.testReaperPostQuiescenceHook = null;
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testGenerationOrderingDuringReclamation() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);

            // Register initial callback X and start it executing on a worker thread
            CountDownLatch xEntered = new CountDownLatch(1);
            CountDownLatch xRelease = new CountDownLatch(1);
            ArgusBackend.setLogCallback((lvl, msg) -> {
                if (msg.contains("Message X")) {
                    xEntered.countDown();
                    try {
                        xRelease.await();
                    } catch (InterruptedException ignored) {}
                }
            });

            Thread workerX = Thread.ofPlatform().start(() -> {
                try (Arena a = Arena.ofConfined()) {
                    ArgusTestBindings.emitLog(1, a.allocateFrom("Message X\n"));
                } catch (Throwable ignored) {}
            });
            assertTrue(xEntered.await(5, TimeUnit.SECONDS), "Callback X should enter");

            // While X is running, replace with G1, then replace with G2
            ArgusBackend.setLogCallback((lvl, msg) -> {}); // X retired (Gen 1)
            ArgusBackend.setLogCallback((lvl, msg) -> {}); // G1 retired (Gen 2)

            // Setup hook: When reaper reaches quiescence for Gen 2, retire G2 by registering G3!
            CountDownLatch hookFired = new CountDownLatch(1);
            ArgusBackend.testReaperPostQuiescenceHook = (eligibleGen) -> {
                if (eligibleGen >= 2 && hookFired.getCount() > 0) {
                    hookFired.countDown();
                    // Concurrently retire G2 by registering G3 (Gen 3)
                    ArgusBackend.setLogCallback((lvl, msg) -> {});
                }
            };

            // Release X so in_flight drops to 0 and reaper observes quiescence for Gen 2
            xRelease.countDown();
            workerX.join(5000);

            assertTrue(hookFired.await(5, TimeUnit.SECONDS), "Hook should fire during quiescence");

            // Give reaper a moment to finish reclaiming up to eligibleGen (Gen 2)
            Thread.sleep(100);

            // G1 and G2 must be reclaimed, and then G3 on subsequent quiescence
            long deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(), "All generations cleanly reclaimed");
        } finally {
            ArgusBackend.testReaperPostQuiescenceHook = null;
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testMultipleActiveConcurrentGenerations() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);

            CountDownLatch aEntered = new CountDownLatch(1);
            CountDownLatch aRelease = new CountDownLatch(1);
            CountDownLatch aDone = new CountDownLatch(1);

            CountDownLatch bEntered = new CountDownLatch(1);
            CountDownLatch bRelease = new CountDownLatch(1);
            CountDownLatch bDone = new CountDownLatch(1);

            // Register Callback A
            ArgusBackend.setLogCallback((lvl, msg) -> {
                if (msg.contains("Log A")) {
                    aEntered.countDown();
                    try {
                        aRelease.await();
                    } catch (InterruptedException ignored) {}
                    aDone.countDown();
                }
            });

            // Start thread 1 emitting Log A
            Thread t1 = Thread.ofPlatform().start(() -> {
                try (Arena a = Arena.ofConfined()) {
                    ArgusTestBindings.emitLog(1, a.allocateFrom("Log A\n"));
                } catch (Throwable ignored) {}
            });
            assertTrue(aEntered.await(5, TimeUnit.SECONDS), "Callback A should enter");

            // While A is running, replace with Callback B (retiring A)
            ArgusBackend.setLogCallback((lvl, msg) -> {
                if (msg.contains("Log B")) {
                    bEntered.countDown();
                    try {
                        bRelease.await();
                    } catch (InterruptedException ignored) {}
                    bDone.countDown();
                }
            });

            // Start thread 2 emitting Log B
            Thread t2 = Thread.ofPlatform().start(() -> {
                try (Arena a = Arena.ofConfined()) {
                    ArgusTestBindings.emitLog(1, a.allocateFrom("Log B\n"));
                } catch (Throwable ignored) {}
            });
            assertTrue(bEntered.await(5, TimeUnit.SECONDS), "Callback B should enter");

            // While both A and B are running, replace with Callback C (retiring B)
            ArgusBackend.setLogCallback((lvl, msg) -> {});

            // Now both A and B are retired, but both are actively executing
            assertEquals(2, ArgusBackend.getRetiredCallbackArenaCount(),
                "Both A and B arenas must be retained while in-flight count > 0");

            // Release A only
            aRelease.countDown();
            t1.join(5000);
            assertTrue(aDone.await(5, TimeUnit.SECONDS), "Callback A should complete");

            // B is STILL running, so neither A nor B should be closed yet because global inFlight > 0
            Thread.sleep(100);
            assertEquals(2, ArgusBackend.getRetiredCallbackArenaCount(),
                "Arenas must remain retained while thread B is still executing");

            // Release B
            bRelease.countDown();
            t2.join(5000);
            assertTrue(bDone.await(5, TimeUnit.SECONDS), "Callback B should complete");

            // Now both are finished; wait for reaper to reclaim all
            long deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "All retired arenas should be cleanly reclaimed once all threads quiesce");
        } finally {
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }

    @Test
    void testReaperQueryFailureSafeRetention() throws Throwable {
        Assumptions.assumeTrue(ArgusTestBindings.hasTestHooks(), "Native test hooks required");
        ArgusBackend.init();
        ArgusLogLevel initial = ArgusBackend.getLogLevel();
        try {
            ArgusBackend.setLogLevel(ArgusLogLevel.DEBUG);

            // Register initial callback
            ArgusBackend.setLogCallback((lvl, msg) -> {});

            // Simulate persistent native query failure BEFORE replacing callback
            AtomicInteger queryAttempts = new AtomicInteger(0);
            ArgusBackend.testInFlightCountOverride = () -> {
                queryAttempts.incrementAndGet();
                throw new RuntimeException("Simulated native query linker failure");
            };

            // Replacing callback now attempts synchronous drain, which fails query and retains arena safely
            ArgusBackend.setLogCallback((lvl, msg) -> {}); // retires previous

            assertEquals(1, ArgusBackend.getRetiredCallbackArenaCount(), "One arena should be retired");

            // Explicitly ensure reaper is scheduled if not already
            ArgusBackend.drainRetiredCallbackArenas();

            // Wait for reaper to fail 3 times and terminate cleanly into safe retention
            long deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.isReaperRunning()) {
                Thread.sleep(20);
            }

            // Invariants:
            // 1. Reaper must have stopped running (bounded retry, no infinite virtual-thread respawning)
            assertFalse(ArgusBackend.isReaperRunning(), "Reaper must terminate after consecutive failures");
            assertTrue(queryAttempts.get() >= 3, "Reaper should have attempted at least 3 queries with backoff");

            // 2. The retired arena MUST NOT have been closed or destroyed (Safe Retention invariant)
            assertEquals(1, ArgusBackend.getRetiredCallbackArenaCount(),
                "Retired arena must be safely retained in memory when quiescence cannot be verified");

            // Restore healthy query operation
            ArgusBackend.testInFlightCountOverride = null;

            // Subsequent drain/reaper now successfully reclaims the arena
            ArgusBackend.drainRetiredCallbackArenas();
            deadline = System.currentTimeMillis() + 3000;
            while (System.currentTimeMillis() < deadline && ArgusBackend.getRetiredCallbackArenaCount() > 0) {
                Thread.sleep(10);
            }
            assertEquals(0, ArgusBackend.getRetiredCallbackArenaCount(),
                "Arena must be reclaimed once healthy query capability is restored");
        } finally {
            ArgusBackend.testInFlightCountOverride = null;
            ArgusBackend.free();
            ArgusBackend.setLogLevel(initial);
        }
    }
}

