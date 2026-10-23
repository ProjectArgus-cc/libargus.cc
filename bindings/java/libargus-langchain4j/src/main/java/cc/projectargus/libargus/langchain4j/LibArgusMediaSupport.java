package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBitmap;
import cc.projectargus.libargus.ArgusMultimodalContext;
import cc.projectargus.libargus.ArgusNativeException;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.UserMessage;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Zero-Trust sandboxed media extractor and loader for {@link ImageContent} and {@link AudioContent}.
 *
 * <p>Security invariants (Option A - Zero-Trust Sandboxed Policy):
 * <ul>
 *   <li>In-memory Base64 payloads are validated against {@code maxMediaBytes} <em>before</em>
 *       Base64 decoding to prevent heap OOM amplification.</li>
 *   <li>Remote network URIs ({@code http://}, {@code https://}, etc.) are unconditionally rejected
 *       to eliminate SSRF and unexpected network I/O.</li>
 *   <li>Local {@code file://} URIs are rejected by default unless confined to an explicit
 *       {@code allowedMediaRoot} (verified via {@link Path#toRealPath()} to defeat {@code ../}
 *       traversal and symlinks) or explicitly enabled via {@code allowLocalFileMedia(true)}.</li>
 * </ul>
 */
final class LibArgusMediaSupport {

    static final long DEFAULT_MAX_MEDIA_BYTES = 64L * 1024L * 1024L; // 64 MiB

    private LibArgusMediaSupport() {}

    static boolean hasMediaContent(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        for (ChatMessage msg : messages) {
            if (msg instanceof UserMessage user) {
                for (Content c : user.contents()) {
                    if (c instanceof ImageContent || c instanceof AudioContent) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static List<ArgusBitmap> extractBitmaps(
        ArgusMultimodalContext mctx,
        List<ChatMessage> messages,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes
    ) {
        if (messages == null || messages.isEmpty()) {
            return Collections.emptyList();
        }
        List<ArgusBitmap> bitmaps = new ArrayList<>();
        try {
            for (ChatMessage msg : messages) {
                if (msg instanceof UserMessage user) {
                    for (Content c : user.contents()) {
                        if (c instanceof ImageContent ic) {
                            Image img = ic.image();
                            if (img == null) {
                                throw new IllegalArgumentException("ImageContent contains null Image");
                            }
                            bitmaps.add(loadMediaItem(
                                mctx,
                                img.base64Data(),
                                img.url(),
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                "ImageContent"
                            ));
                        } else if (c instanceof AudioContent ac) {
                            Audio audio = ac.audio();
                            if (audio == null) {
                                throw new IllegalArgumentException("AudioContent contains null Audio");
                            }
                            bitmaps.add(loadMediaItem(
                                mctx,
                                audio.base64Data(),
                                audio.url(),
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                "AudioContent"
                            ));
                        }
                    }
                }
            }
            return bitmaps;
        } catch (Throwable t) {
            closeAllQuietly(bitmaps);
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to extract multimodal media content", t);
        }
    }

    static void closeAllQuietly(List<ArgusBitmap> bitmaps) {
        if (bitmaps == null) {
            return;
        }
        for (ArgusBitmap bmp : bitmaps) {
            if (bmp != null) {
                try {
                    bmp.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private static ArgusBitmap loadMediaItem(
        ArgusMultimodalContext mctx,
        String base64Data,
        URI uri,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        String contentKind
    ) {
        if (base64Data != null && !base64Data.isEmpty()) {
            return loadFromBase64(mctx, base64Data, maxMediaBytes, contentKind);
        }
        if (uri != null) {
            return loadFromUri(mctx, uri, allowedMediaRoot, allowLocalFileMedia, maxMediaBytes, contentKind);
        }
        throw new IllegalArgumentException(contentKind + " must specify either non-empty base64Data or a valid file:// URI");
    }

    private static ArgusBitmap loadFromBase64(
        ArgusMultimodalContext mctx,
        String rawBase64,
        long maxMediaBytes,
        String contentKind
    ) {
        String payload = rawBase64.trim();
        int commaIdx = payload.indexOf(',');
        if (commaIdx >= 0 && payload.substring(0, commaIdx).toLowerCase(Locale.ROOT).contains("base64")) {
            payload = payload.substring(commaIdx + 1).trim();
        }
        // Pre-decode size bound check to prevent heap OOM amplification
        long estimatedDecodedBytes = ((long) payload.length() * 3L) / 4L;
        if (estimatedDecodedBytes > maxMediaBytes) {
            throw new IllegalArgumentException(
                contentKind + " Base64 payload (~" + estimatedDecodedBytes +
                " bytes) exceeds maxMediaBytes limit (" + maxMediaBytes + " bytes)"
            );
        }
        byte[] decoded = Base64.getMimeDecoder().decode(payload);
        if (decoded.length == 0) {
            throw new IllegalArgumentException(contentKind + " Base64 payload decoded to 0 bytes");
        }
        if (decoded.length > maxMediaBytes) {
            throw new IllegalArgumentException(
                contentKind + " decoded payload (" + decoded.length +
                " bytes) exceeds maxMediaBytes limit (" + maxMediaBytes + " bytes)"
            );
        }
        ArgusBitmap bmp = ArgusBitmap.loadBuffer(mctx, decoded, false);
        if (bmp == null) {
            throw new ArgusNativeException(
                ArgusNativeException.ERROR_INVALID_ARGUMENT,
                "Failed to decode " + contentKind + " buffer via ArgusBitmap.loadBuffer"
            );
        }
        return bmp;
    }

    private static ArgusBitmap loadFromUri(
        ArgusMultimodalContext mctx,
        URI uri,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        String contentKind
    ) {
        String scheme = uri.getScheme();
        if (scheme == null || !scheme.equalsIgnoreCase("file")) {
            throw new IllegalArgumentException(
                "Disallowed URI scheme '" + scheme + "' in " + contentKind +
                ": remote network media fetches are prohibited (SSRF defense). Provide Base64 data or an authorized file:// URI."
            );
        }

        if (allowedMediaRoot == null && !allowLocalFileMedia) {
            throw new IllegalArgumentException(
                "Local file:// media URIs are disabled by default for security in " + contentKind +
                "; configure .allowedMediaRoot(Path) or .allowLocalFileMedia(true) on the model builder, or pass Base64 media content."
            );
        }

        final Path realFilePath;
        try {
            realFilePath = Path.of(uri).toRealPath();
        } catch (IOException | IllegalArgumentException e) {
            throw new IllegalArgumentException("Failed to resolve local media file URI in " + contentKind + ": " + uri, e);
        }

        if (allowedMediaRoot != null) {
            final Path realRoot;
            try {
                realRoot = allowedMediaRoot.toRealPath();
            } catch (IOException e) {
                throw new IllegalArgumentException("Failed to resolve allowedMediaRoot: " + allowedMediaRoot, e);
            }
            if (!realFilePath.startsWith(realRoot)) {
                throw new SecurityException(
                    "Local media file '" + realFilePath + "' escapes configured allowedMediaRoot '" + realRoot + "'"
                );
            }
        }

        if (!Files.isRegularFile(realFilePath)) {
            throw new IllegalArgumentException(
                contentKind + " URI must point to a regular file: " + realFilePath
            );
        }

        final long fileSize;
        try {
            fileSize = Files.size(realFilePath);
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to read size of local media file: " + realFilePath, e);
        }
        if (fileSize <= 0 || fileSize > maxMediaBytes) {
            throw new IllegalArgumentException(
                contentKind + " file size (" + fileSize + " bytes) must be in [1, " + maxMediaBytes + "]"
            );
        }

        ArgusBitmap bmp = ArgusBitmap.loadFile(mctx, realFilePath, false);
        if (bmp == null) {
            throw new ArgusNativeException(
                ArgusNativeException.ERROR_INVALID_ARGUMENT,
                "Failed to load " + contentKind + " file via ArgusBitmap.loadFile: " + realFilePath
            );
        }
        return bmp;
    }
}
