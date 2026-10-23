# Android & JDK 17+ JNI Integration Guide (`libargus-jni`)

For applications targeting **Android (API 26+ / `arm64-v8a` / `x86_64`)** or **JDK 17–21 JVMs** where Project Panama FFM (`java.lang.foreign.*`) is unavailable, `libargus-jni` provides a 100% feature-complete, package-compatible (`cc.projectargus.libargus.*`) drop-in binding.

---

## 1. Dependency & Native Binary Setup

### 1.1 Desktop / Server JDK 17–21 (`build.gradle.kts`)
On Linux, Windows, and macOS running JDK 17+, pair `libargus-jni` with one of the precompiled `--release 17` platform SPI runtime JARs:

```kotlin
dependencies {
    implementation("cc.projectargus:libargus-jni:2.3.1")

    // Automatic SPI extraction (choose cpu, cuda, rocm, vulkan, or macos-metal):
    runtimeOnly("cc.projectargus:libargus-native-linux-cpu:2.3.1")
}
```

### 1.2 Android Application / Library Module (`build.gradle.kts`)
On Android, add `libargus-jni` as an `implementation` dependency and place the cross-compiled `libargus.so` inside `src/main/jniLibs/<abi>/libargus.so` (e.g. `src/main/jniLibs/arm64-v8a/libargus.so`):

```kotlin
dependencies {
    implementation("cc.projectargus:libargus-jni:2.3.1")
}
```

Because `libargus` compiles both the C ABI and `Java_cc_projectargus_libargus_jni_ArgusJni_*` JNI symbols into the single `libargus.so` binary, Android's Bionic linker resolves `System.loadLibrary("argus")` directly from your APK's `jniLibs` directory with zero namespace collisions or temporary file extraction.

> [!NOTE]
> **Custom Extraction Directory on Restricted Filesystems:** If you load via SPI on a container or embedded Linux environment where `/tmp` (`java.io.tmpdir`) is mounted `noexec`, set `-Dcc.projectargus.libargus.nativeDir=/path/to/exec/cache` (e.g. `context.getCodeCacheDir().getAbsolutePath()` on Android) or export `LIBARGUS_NATIVE_DIR` before referencing `ArgusBackend`.

---

## 2. Safe File Descriptor Model Loading on Android (`ArgusModel.loadFd`)

On Android 10+ (Scoped Storage), models selected via the Storage Access Framework (`content://` URIs) or bundled as uncompressed APK assets (`AssetFileDescriptor`) are accessed via `ParcelFileDescriptor`.

`ArgusModel.loadFd(int fd, int gpuLayers, boolean useMlock)` loads GGUF weights directly from an open POSIX file descriptor:

```java
import android.content.ContentResolver;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import cc.projectargus.libargus.*;

public void runInferenceFromContentUri(ContentResolver resolver, Uri modelUri) throws Exception {
    ArgusBackend.init();

    // IMPORTANT: Use try-with-resources and pfd.getFd() (NOT pfd.detachFd()!)
    try (ParcelFileDescriptor pfd = resolver.openFileDescriptor(modelUri, "r")) {
        if (pfd == null) {
            throw new IllegalStateException("Unable to open Content URI: " + modelUri);
        }

        try (ArgusModel model = ArgusModel.loadFd(pfd.getFd(), 99, false);
             ArgusContext context = ArgusContext.init(model, new ArgusContextConfig.Builder(4096).build())) {

            ArgusSamplerConfig sampler = new ArgusSamplerConfig.Builder().temperature(0.6f).build();
            String response = context.generate("Hello from Android JNI!", 128, sampler, System.out::print);
        }
    }
}
```

> [!WARNING]
> **File Descriptor Ownership Contract (`pfd.getFd()` vs. `pfd.detachFd()`):**
> Native `argus_model_load_fd` **duplicates** the passed descriptor (`dup(fd)`), rewinds the stream to offset `0` (`fseek(fp, 0, SEEK_SET)`), and closes its duplicated `FILE*` handle when weight loading completes. **The caller retains ownership of the original `fd`.**
> Always pass `pfd.getFd()` inside a `try (ParcelFileDescriptor pfd = ...)` block so `ParcelFileDescriptor.close()` closes the original descriptor cleanly. Calling `pfd.detachFd()` without manually calling `Os.close(fd)` will leak a file descriptor on every load.

---

## 3. Zero-Copy Off-Heap JNI Overloads (`DirectByteBuffer`)

To avoid JNI primitive array copy overhead on steady-state inference, audio capture, and camera frame ingestion, `libargus-jni` provides `*Direct` methods backed by JNI `GetDirectBufferAddress`:

| Subsystem | Zero-Copy JNI Method | Buffer Alignment & Layout |
| :--- | :--- | :--- |
| **Token Decoding** | `ArgusContext.decodeBatchDirect(ByteBuffer, int nTokens, int startPos, int seqId, boolean logits, ArgusAbortFlag)` | Native-endian `int32` tokens; `position()` must be 4-byte aligned. |
| **Embedding Evaluation** | `ArgusContext.evalEmbeddingsDirect(ByteBuffer tokensBuf, int nTokens, int seqId, ByteBuffer outFloatBuf)` | Native-endian `int32` input & writable `float32` output; both 4-byte aligned. |
| **Embedding Copy-Out** | `ArgusContext.getEmbeddingsDirect(int seqId, ByteBuffer outFloatBuf, int maxFloats)` | Writable direct `ByteBuffer` (`float32`), 4-byte aligned. |
| **Whisper ASR** | `ArgusAudioContext.transcribeDirect(ByteBuffer pcmDirectBuf, int sampleCount, int maxChars)` | 16 kHz `float32` PCM samples; `position()` must be 4-byte aligned. |
| **Vision RGB Ingestion** | `ArgusBitmap.fromRgbDirect(int width, int height, ByteBuffer rgbDirectBuf)` | Packed 3-channel `RGBRGB...` bytes (`width * height * 3` bytes). |
| **Multimodal Audio** | `ArgusBitmap.fromPcmDirect(ByteBuffer pcmDirectBuf, int nSamples)` | `float32` PCM samples; `position()` must be 4-byte aligned. |

```java
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

// Allocate a reusable off-heap direct buffer aligned to native endianness
ByteBuffer tokenBuf = ByteBuffer.allocateDirect(512 * Integer.BYTES).order(ByteOrder.nativeOrder());
tokenBuf.asIntBuffer().put(0, model.vocabBos());
tokenBuf.position(0).limit(Integer.BYTES);

context.decodeBatchDirect(tokenBuf, 1, 0, 0, true, null);
```

---

## 4. Mobile Thermal & Power Governor Integration (`setNThreads`)

Mobile SoCs (ARM big.LITTLE / DynamIQ) experience thermal throttling or background scheduling transitions during long conversations. You can dynamically scale single-token generation threads (`nThreads`) and prompt batch prefill threads (`nThreadsBatch`) on an active `ArgusContext` or `ArgusAudioContext` without purging the KV cache:

```java
// Scale up to performance cores during interactive foreground prefill:
context.setNThreads(4, 8);

// Scale down when thermal state rises or app enters split-screen / background:
context.setNThreads(2, 2);
audioContext.setNThreads(2);
```

---

## 5. ProGuard / R8 Rules for Android Release Builds

`libargus-jni` resolves JNI native methods on `cc.projectargus.libargus.internal.ArgusJni` and dispatches native log callbacks to `ArgusJni.onNativeLog(int, byte[])`. Add the following rule to your `proguard-rules.pro`:

```proguard
-keep class cc.projectargus.libargus.** { *; }
```
