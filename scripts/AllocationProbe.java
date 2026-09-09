import cc.projectargus.libargus.*;
import java.lang.foreign.*;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;

/** Manual allocation measurement, not a timing/coverage assertion. JDK 22+. */
class AllocationProbe {
    public static void main(String[] args) {
        var meter = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        if (!meter.isThreadAllocatedMemorySupported()) throw new IllegalStateException("Heap allocation counters unavailable");
        meter.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().threadId();
        ArgusBackend.init();
        try (Arena arena = Arena.ofConfined();
             ArgusModel model = ArgusModel.load(arena, Path.of(args[0], "tiny.gguf"), 0, false);
             ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128));
             ArgusMultimodalContext mctx = ArgusMultimodalContext.init(arena, model, Path.of(args[0], "tiny-mmproj.gguf"), 2, false)) {
            MemorySegment token = arena.allocateFrom(ValueLayout.JAVA_INT, 1);
            long decode = 0, sample = 0;
            int iterations = 2000;
            for (int i = -2000; i < iterations; i++) {
                ctx.clearCacheSlot(0, -1, -1);
                long before = meter.getThreadAllocatedBytes(thread);
                int result = ctx.decodeBatch(token, 1, 0, 0, true);
                long decoded = meter.getThreadAllocatedBytes(thread);
                int selected = ctx.sampleToken(0, 0, 1);
                long sampled = meter.getThreadAllocatedBytes(thread);
                if (result != 0 || selected < 0) throw new AssertionError("Probe decode/sample failed");
                if (i >= 0) { decode += decoded - before; sample += sampled - decoded; }
            }
            long videoBytes = 0; int reads = 0;
            for (int iteration = -20; iteration < 20; iteration++) {
                try (ArgusVideo video = ArgusVideo.loadFile(arena, mctx, Path.of(args[0], "tiny-video.y4m"), 2, 0)) {
                    for (;;) {
                        long before = meter.getThreadAllocatedBytes(thread);
                        var item = video.readNext();
                        long after = meter.getThreadAllocatedBytes(thread);
                        if (iteration >= 0) { videoBytes += after - before; reads++; }
                        if (item == null) break;
                        if (item.bitmap() != null) item.bitmap().close();
                    }
                }
            }
            System.out.printf("ARGUS_ALLOCATION jdk=%s decode_java_bytes_per_call=%.2f sample_java_bytes_per_call=%.2f video_java_bytes_per_read=%.2f native_allocations=not_measured%n",
                System.getProperty("java.version"), (double)decode/iterations, (double)sample/iterations, (double)videoBytes/reads);
        } finally { ArgusBackend.free(); }
    }
}
