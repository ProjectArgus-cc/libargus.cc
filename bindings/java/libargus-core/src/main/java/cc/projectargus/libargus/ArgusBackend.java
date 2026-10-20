package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBackendSupport;
import cc.projectargus.libargus.internal.ArgusBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.IntSupplier;

/**
 * Handles the process-global compute backends (CUDA, CPU, Metal) registry and telemetry.
 */
public final class ArgusBackend {
    public static final long FEATURE_CPU_ACCEL = ArgusBackendSupport.FEATURE_CPU_ACCEL;
    public static final long FEATURE_CPU       = ArgusBackendSupport.FEATURE_CPU;
    public static final long FEATURE_CUDA      = ArgusBackendSupport.FEATURE_CUDA;
    public static final long FEATURE_ROCM      = ArgusBackendSupport.FEATURE_ROCM;
    public static final long FEATURE_VULKAN    = ArgusBackendSupport.FEATURE_VULKAN;
    public static final long FEATURE_METAL     = ArgusBackendSupport.FEATURE_METAL;
    public static final long FEATURE_SYCL      = ArgusBackendSupport.FEATURE_SYCL;

    private static boolean initialized = false;

    private ArgusBackend() {}

    public static synchronized boolean init() {
        return init(null);
    }

    /**
     * Initializes the process-global hardware execution registry with a custom backend plugin path.
     * Must be called once before loading models or executing transcription/speech.
     *
     * @param customPluginPath explicit directory to search for dynamic ggml plugin libraries, or null to use the compiled backends
     * @return true if backend registers successfully, false otherwise.
     */
    public static synchronized boolean init(String customPluginPath) {
        if (initialized) {
            return true;
        }
        drainRetiredArenas();
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment pathSeg = (customPluginPath != null && !customPluginPath.isEmpty())
                ? localArena.allocateFrom(customPluginPath)
                : MemorySegment.NULL;
            initialized = (boolean) ArgusBindings.argus_backend_init.invokeExact(pathSeg);
            return initialized;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_init", t);
        }
    }

    /**
     * Queries whether the process-global native compute backend registry is currently initialized.
     *
     * @return true if backend is initialized and active, false otherwise
     */
    public static boolean isInitialized() {
        try {
            return (boolean) ArgusBindings.argus_backend_is_initialized.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_is_initialized", t);
        }
    }

    /**
     * Queries the compile-time feature bitmask representing backends and acceleration built into the native binary.
     *
     * @return bitmask combining FEATURE_* flags
     */
    public static long getBuildFeatures() {
        try {
            return (long) ArgusBindings.argus_build_features.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_build_features", t);
        }
    }

    /**
     * Checks if a specific feature flag is present in the native library build.
     *
     * @param featureMask feature flag bitmask (e.g. FEATURE_CPU_ACCEL, FEATURE_CUDA)
     * @return true if enabled at compile time
     */
    public static boolean hasFeature(long featureMask) {
        return (getBuildFeatures() & featureMask) != 0;
    }

    /**
     * Deallocates global backend context registration and frees active compute pools.
     */
    public static synchronized void free() {
        if (!initialized) {
            return;
        }
        try {
            try {
                ArgusBindings.argus_set_log_callback.invokeExact(MemorySegment.NULL, MemorySegment.NULL);
            } catch (Throwable ignored) {}

            if (logCallbackArena != null) {
                long gen = RETIREMENT_SEQUENCE.incrementAndGet();
                retiredCallbacks.add(new RetiredCallback(gen, logCallbackArena));
                logCallbackArena = null;
            }

            boolean reentrant = Boolean.TRUE.equals(IN_LOG_CALLBACK.get());
            if (!reentrant) {
                long deadline = System.currentTimeMillis() + 1000;
                while (System.currentTimeMillis() < deadline) {
                    try {
                        int inFlight = getInFlightCount();
                        if (inFlight == 0) {
                            break;
                        }
                    } catch (Throwable ignored) {
                        break;
                    }
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }

                drainRetiredArenas();
            } else {
                scheduleReaperIfNeeded();
            }

            ArgusBindings.argus_backend_free.invokeExact();
            initialized = false;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_free", t);
        }
    }

    record RetiredCallback(long generation, Arena arena) {}

    private static Arena logCallbackArena = null;
    private static final AtomicLong RETIREMENT_SEQUENCE = new AtomicLong(0);
    private static final ConcurrentLinkedQueue<RetiredCallback> retiredCallbacks = new ConcurrentLinkedQueue<>();
    private static final Object RECLAMATION_LOCK = new Object();
    private static final AtomicBoolean REAPER_RUNNING = new AtomicBoolean(false);

    // Testing hooks for deterministic synchronization and fault injection
    static volatile Consumer<Long> testReaperPostQuiescenceHook = null;
    static volatile IntSupplier testInFlightCountOverride = null;
    static volatile long testReaperTimeoutMs = -1;

    private static int getInFlightCount() throws Throwable {
        var override = testInFlightCountOverride;
        if (override != null) {
            return override.getAsInt();
        }
        return (int) ArgusBindings.argus_log_in_flight_count.invokeExact();
    }

    private static void scheduleReaperIfNeeded() {
        if (retiredCallbacks.isEmpty()) {
            return;
        }
        if (REAPER_RUNNING.compareAndSet(false, true)) {
            Thread.ofVirtual().name("argus-callback-reaper").start(() -> {
                int consecutiveFailures = 0;
                long timeout = testReaperTimeoutMs > 0 ? testReaperTimeoutMs : 10_000;
                long deadline = System.currentTimeMillis() + timeout;
                try {
                    while (System.currentTimeMillis() < deadline && !retiredCallbacks.isEmpty()) {
                        long eligibleGen = RETIREMENT_SEQUENCE.get();
                        int inFlight;
                        try {
                            inFlight = getInFlightCount();
                            consecutiveFailures = 0;
                        } catch (Throwable t) {
                            consecutiveFailures++;
                            if (consecutiveFailures >= 3) {
                                // Query failure barrier: safe retention without task churn
                                return;
                            }
                            try {
                                Thread.sleep(50L * consecutiveFailures);
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                return;
                            }
                            continue;
                        }

                        if (inFlight == 0) {
                            var hook = testReaperPostQuiescenceHook;
                            if (hook != null) {
                                hook.accept(eligibleGen);
                            }
                            reclaimGenerationsUpTo(eligibleGen);
                            if (retiredCallbacks.isEmpty()) {
                                return;
                            }
                        }

                        try {
                            Thread.sleep(10);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                } finally {
                    REAPER_RUNNING.set(false);
                    if (!retiredCallbacks.isEmpty() && consecutiveFailures == 0) {
                        if (System.currentTimeMillis() < deadline) {
                            scheduleReaperIfNeeded();
                        } else {
                            // Normal 10-second deadline elapsed with pending work and healthy native queries.
                            // Schedule a bounded delayed successor to guarantee eventual quiescence and reclamation.
                            Thread.ofVirtual().name("argus-callback-reaper-handoff").start(() -> {
                                try {
                                    Thread.sleep(100);
                                } catch (InterruptedException e) {
                                    Thread.currentThread().interrupt();
                                    return;
                                }
                                scheduleReaperIfNeeded();
                            });
                        }
                    }
                }
            });
        }
    }

    private static void reclaimGenerationsUpTo(long eligibleGen) {
        synchronized (RECLAMATION_LOCK) {
            RetiredCallback item;
            while ((item = retiredCallbacks.peek()) != null) {
                if (item.generation() > eligibleGen) {
                    break;
                }
                retiredCallbacks.poll();
                try {
                    item.arena().close();
                } catch (Throwable ignored) {}
            }
        }
    }

    private static void drainRetiredArenas() {
        if (retiredCallbacks.isEmpty()) {
            return;
        }
        long eligibleGen = RETIREMENT_SEQUENCE.get();
        int inFlight = -1;
        try {
            inFlight = getInFlightCount();
        } catch (Throwable ignored) {
            // Cannot verify quiescence synchronously; delegate to reaper for bounded retry & safe retention
        }
        if (inFlight == 0) {
            reclaimGenerationsUpTo(eligibleGen);
        } else {
            scheduleReaperIfNeeded();
        }
    }

    static int getRetiredCallbackArenaCount() {
        return retiredCallbacks.size();
    }

    static void drainRetiredCallbackArenas() {
        drainRetiredArenas();
    }

    static boolean isReaperRunning() {
        return REAPER_RUNNING.get();
    }

    /**
     * Sets the global minimum native log severity threshold.
     * Messages below this severity tier are discarded with zero I/O overhead.
     *
     * @param level minimum severity tier to process. Set to {@link ArgusLogLevel#NONE} to silence completely.
     */
    public static void setLogLevel(ArgusLogLevel level) {
        if (level == null) {
            throw new IllegalArgumentException("Log level cannot be null");
        }
        try {
            ArgusBindings.argus_set_log_level.invokeExact(level.getValue());
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_set_log_level", t);
        }
    }

    /**
     * Queries the active global minimum native log severity threshold.
     *
     * @return current log level threshold
     */
    public static ArgusLogLevel getLogLevel() {
        try {
            int val = (int) ArgusBindings.argus_get_log_level.invokeExact();
            return ArgusLogLevel.fromValue(val);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_get_log_level", t);
        }
    }

    /**
     * Convenience method to silence or restore standard warning logging.
     *
     * @param quiet true to set level to NONE, false to restore WARN
     */
    public static void setQuiet(boolean quiet) {
        setLogLevel(quiet ? ArgusLogLevel.NONE : ArgusLogLevel.WARN);
    }

    /**
     * Registers a custom log handler callback for native diagnostic telemetry.
     *
     * @param callback target handler receiving messages, or null to restore default stderr formatting
     */
    public static synchronized void setLogCallback(ArgusLogCallback callback) {
        try {
            if (callback == null) {
                ArgusBindings.argus_set_log_callback.invokeExact(MemorySegment.NULL, MemorySegment.NULL);
                if (logCallbackArena != null) {
                    long gen = RETIREMENT_SEQUENCE.incrementAndGet();
                    retiredCallbacks.add(new RetiredCallback(gen, logCallbackArena));
                    logCallbackArena = null;
                }
                drainRetiredArenas();
                return;
            }

            MethodHandle baseHandle = MethodHandles.lookup().findStatic(
                ArgusBackend.class,
                "dispatchNativeLog",
                MethodType.methodType(void.class, ArgusLogCallback.class, int.class, MemorySegment.class, MemorySegment.class)
            );
            MethodHandle boundHandle = MethodHandles.insertArguments(baseHandle, 0, callback);

            Arena newArena = Arena.ofShared();
            MemorySegment stub = Linker.nativeLinker().upcallStub(
                boundHandle,
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS),
                newArena
            );

            ArgusBindings.argus_set_log_callback.invokeExact(stub, MemorySegment.NULL);

            if (logCallbackArena != null) {
                long gen = RETIREMENT_SEQUENCE.incrementAndGet();
                retiredCallbacks.add(new RetiredCallback(gen, logCallbackArena));
            }
            logCallbackArena = newArena;
            drainRetiredArenas();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error registering native log callback", t);
        }
    }

    private static final ThreadLocal<Boolean> IN_LOG_CALLBACK = ThreadLocal.withInitial(() -> Boolean.FALSE);

    private static void dispatchNativeLog(ArgusLogCallback cb, int level, MemorySegment textPtr, MemorySegment userDataPtr) {
        if (cb == null || textPtr.equals(MemorySegment.NULL)) {
            return;
        }
        IN_LOG_CALLBACK.set(Boolean.TRUE);
        try {
            String msg = textPtr.reinterpret(Long.MAX_VALUE).getString(0);
            cb.onLog(ArgusLogLevel.fromValue(level), msg);
        } catch (Throwable ignored) {
            // Exception containment barrier: protect native callers from upcall exceptions
        } finally {
            IN_LOG_CALLBACK.set(Boolean.FALSE);
            drainRetiredArenas();
        }
    }

    /**
     * @return count of available hardware backends.
     */
    public static int getCount() {
        try {
            return (int) ArgusBindings.argus_backend_get_count.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_get_count", t);
        }
    }

    /**
     * Query driver registry name at index.
     * @param index backend index.
     * @return dynamic backend name (e.g. "CUDA0", "CPU").
     */
    public static String getName(int index) {
        try {
            MemorySegment namePtr = (MemorySegment) ArgusBindings.argus_backend_get_name.invokeExact(index);
            if (namePtr.equals(MemorySegment.NULL)) {
                return "UNKNOWN";
            }
            return namePtr.reinterpret(256).getString(0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Fatal error running argus_backend_get_name", t);
        }
    }

    /**
     * Utility listing all available backends on the machine.
     * @return list of backend driver names.
     */
    public static List<String> getAvailableBackends() {
        int count = getCount();
        if (count <= 0) {
            return Collections.emptyList();
        }
        List<String> list = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            list.add(getName(i));
        }
        return Collections.unmodifiableList(list);
    }

    /**
     * Returns the compiled framework version string.
     */
    public static String getVersion() {
        return ArgusBindings.VERSION;
    }

    /**
     * Isolated CLI entry point verifying SPI native library extraction and build feature masks.
     *
     * @param args optional args[0] specifies expected backend target: "cpu", "cuda", "rocm", "vulkan", "metal"
     */
    public static void main(String[] args) {
        ArgusBackendSupport.runCliProbe("ArgusBackend", getVersion(), ArgusBindings.EXTRACTED_DIR, getBuildFeatures(), args);
    }

    public static long expectedFeatureMask(String target) {
        return ArgusBackendSupport.expectedFeatureMask(target);
    }

    /** Build metadata only; does not initialize drivers or perform device execution. */
    public static java.util.Map<String, String> getBuildInfo() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(4096);
            int required = (int) ArgusBindings.argus_build_info_copy.invokeExact(out, 4096);
            if (required < 0 || required >= 4096) throw new IllegalStateException("Invalid build metadata size");
            java.util.Map<String, String> info = ArgusBackendSupport.parseBuildInfo(out.getString(0), getBuildFeatures());
            if (!cc.projectargus.libargus.internal.ArgusLayouts.diagnosticAbi().equals(info.get("abi_layouts")))
                throw new IllegalStateException("Native ABI layout differs from Java layouts");
            return java.util.Map.copyOf(info);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new IllegalStateException("Cannot read native build metadata", t);
        }
    }
}
