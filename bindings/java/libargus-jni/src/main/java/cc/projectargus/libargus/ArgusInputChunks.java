package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusJni;
import cc.projectargus.libargus.internal.ArgusJniResource;

/**
 * Holds tokenized prompt chunk lists in unmanaged native memory.
 */
public final class ArgusInputChunks extends ArgusJniResource {

    private ArgusInputChunks(long chunksHandle) {
        super(chunksHandle);
    }

    @Override
    protected String resourceName() {
        return "ArgusInputChunks";
    }

    @Override
    protected void releaseNativeHandle(long oldHandle) {
        ArgusJni.nativeInputChunksFree(oldHandle);
    }

    public static ArgusInputChunks init() {
        long ptr = ArgusJni.nativeInputChunksInit();
        if (ptr == 0L) {
            ArgusNativeException.checkStatus(-1, "argus_input_chunks_init");
        }
        return new ArgusInputChunks(ptr);
    }
}
