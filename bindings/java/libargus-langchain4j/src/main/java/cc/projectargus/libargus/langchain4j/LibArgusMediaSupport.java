package cc.projectargus.libargus.langchain4j;

import cc.projectargus.libargus.ArgusBitmap;
import cc.projectargus.libargus.ArgusMultimodalContext;
import cc.projectargus.libargus.ArgusNativeException;
import cc.projectargus.libargus.ArgusVideo;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;
import dev.langchain4j.data.video.Video;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Zero-Trust sandboxed media extractor and loader for {@link ImageContent}, {@link AudioContent},
 * {@link VideoContent}, and standalone {@link Audio} transcription payloads.
 *
 * <p>Security invariants (Zero-Trust Sandboxed Policy):
 * <ul>
 *   <li>In-memory binary and Base64 payloads are validated against {@code maxMediaBytes} <em>before</em>
 *       decoding to prevent heap OOM amplification.</li>
 *   <li>Remote network URIs ({@code http://}, {@code https://}, etc.) are unconditionally rejected
 *       to eliminate SSRF and unexpected network I/O.</li>
 *   <li>Local {@code file://} URIs are rejected by default unless confined to an explicit
 *       {@code allowedMediaRoot} (verified via {@link Path#toRealPath()} to defeat {@code ../}
 *       traversal and symlinks) or explicitly enabled via {@code allowLocalFileMedia(true)}.</li>
 *   <li>Video frame extraction enforces {@code maxVideoFrames} so multi-hour inputs cannot exhaust
 *       native projector memory or context token budgets.</li>
 * </ul>
 */
final class LibArgusMediaSupport {

    static final long DEFAULT_MAX_MEDIA_BYTES = 64L * 1024L * 1024L; // 64 MiB
    static final float DEFAULT_VIDEO_FPS = 1.0f;
    static final long DEFAULT_VIDEO_TIMESTAMP_INTERVAL_MS = 1_000L;
    static final int DEFAULT_MAX_VIDEO_FRAMES = 16;

    record PreparedMultimodalInput(String prompt, List<ArgusBitmap> bitmaps) {}

    private LibArgusMediaSupport() {}

    static boolean hasMediaContent(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        for (ChatMessage msg : messages) {
            if (msg instanceof UserMessage user) {
                for (Content c : user.contents()) {
                    if (c instanceof ImageContent || c instanceof AudioContent || c instanceof VideoContent) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static boolean hasVideoContent(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        for (ChatMessage msg : messages) {
            if (msg instanceof UserMessage user) {
                for (Content c : user.contents()) {
                    if (c instanceof VideoContent) {
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
        return extractBitmaps(
            mctx,
            messages,
            allowedMediaRoot,
            allowLocalFileMedia,
            maxMediaBytes,
            DEFAULT_VIDEO_FPS,
            DEFAULT_VIDEO_TIMESTAMP_INTERVAL_MS,
            DEFAULT_MAX_VIDEO_FRAMES
        );
    }

    static List<ArgusBitmap> extractBitmaps(
        ArgusMultimodalContext mctx,
        List<ChatMessage> messages,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        float videoFps,
        long videoTimestampIntervalMs,
        int maxVideoFrames
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
                            bitmaps.add(loadBitmapItem(
                                mctx,
                                null,
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
                            bitmaps.add(loadBitmapItem(
                                mctx,
                                audio.binaryData(),
                                audio.base64Data(),
                                audio.url(),
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                "AudioContent"
                            ));
                        } else if (c instanceof VideoContent vc) {
                            readVideoFramesInto(
                                mctx,
                                vc,
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                videoFps,
                                videoTimestampIntervalMs,
                                maxVideoFrames,
                                bitmaps
                            );
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

    static PreparedMultimodalInput prepareMultimodalInput(
        ArgusMultimodalContext mctx,
        String formattedPrompt,
        List<ChatMessage> messages,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        float videoFps,
        long videoTimestampIntervalMs,
        int maxVideoFrames
    ) {
        Objects.requireNonNull(formattedPrompt, "formattedPrompt");
        if (!hasVideoContent(messages)) {
            List<ArgusBitmap> bitmaps = extractBitmaps(
                mctx,
                messages,
                allowedMediaRoot,
                allowLocalFileMedia,
                maxMediaBytes
            );
            return new PreparedMultimodalInput(formattedPrompt, bitmaps);
        }

        List<ArgusBitmap> bitmaps = new ArrayList<>();
        List<String> slotReplacements = new ArrayList<>();
        try {
            for (ChatMessage msg : messages) {
                if (msg instanceof UserMessage user) {
                    for (Content c : user.contents()) {
                        if (c instanceof ImageContent ic) {
                            Image img = ic.image();
                            if (img == null) {
                                throw new IllegalArgumentException("ImageContent contains null Image");
                            }
                            bitmaps.add(loadBitmapItem(
                                mctx,
                                null,
                                img.base64Data(),
                                img.url(),
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                "ImageContent"
                            ));
                            slotReplacements.add(LibArgusChatFormatter.MEDIA_MARKER);
                        } else if (c instanceof AudioContent ac) {
                            Audio audio = ac.audio();
                            if (audio == null) {
                                throw new IllegalArgumentException("AudioContent contains null Audio");
                            }
                            bitmaps.add(loadBitmapItem(
                                mctx,
                                audio.binaryData(),
                                audio.base64Data(),
                                audio.url(),
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                "AudioContent"
                            ));
                            slotReplacements.add(LibArgusChatFormatter.MEDIA_MARKER);
                        } else if (c instanceof VideoContent vc) {
                            String slotExpansion = readVideoFramesInto(
                                mctx,
                                vc,
                                allowedMediaRoot,
                                allowLocalFileMedia,
                                maxMediaBytes,
                                videoFps,
                                videoTimestampIntervalMs,
                                maxVideoFrames,
                                bitmaps
                            );
                            slotReplacements.add(slotExpansion);
                        }
                    }
                }
            }
            String expandedPrompt = expandMediaSlots(formattedPrompt, slotReplacements);
            return new PreparedMultimodalInput(expandedPrompt, bitmaps);
        } catch (Throwable t) {
            closeAllQuietly(bitmaps);
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to prepare multimodal video/media content", t);
        }
    }

    /**
     * Replaces each sequential {@link LibArgusChatFormatter#MEDIA_MARKER} slot in {@code formattedPrompt}
     * with the corresponding entry in {@code slotReplacements}.
     */
    static String expandMediaSlots(String formattedPrompt, List<String> slotReplacements) {
        Objects.requireNonNull(formattedPrompt, "formattedPrompt");
        Objects.requireNonNull(slotReplacements, "slotReplacements");
        final String marker = LibArgusChatFormatter.MEDIA_MARKER;
        final int markerLen = marker.length();

        StringBuilder sb = new StringBuilder(formattedPrompt.length() + slotReplacements.size() * 24);
        int cursor = 0;
        int slotIdx = 0;
        while (true) {
            int pos = formattedPrompt.indexOf(marker, cursor);
            if (pos < 0) {
                sb.append(formattedPrompt, cursor, formattedPrompt.length());
                break;
            }
            if (slotIdx >= slotReplacements.size()) {
                throw new IllegalArgumentException(
                    "Formatted multimodal prompt contains more '" + marker +
                    "' markers than media items (" + slotReplacements.size() + ")"
                );
            }
            sb.append(formattedPrompt, cursor, pos);
            sb.append(slotReplacements.get(slotIdx++));
            cursor = pos + markerLen;
        }
        if (slotIdx != slotReplacements.size()) {
            throw new IllegalArgumentException(
                "Formatted multimodal prompt contains " + slotIdx + " '" + marker +
                "' marker(s), expected " + slotReplacements.size() + " for multimodal media content"
            );
        }
        return sb.toString();
    }

    /**
     * Loads raw audio bytes from {@link Audio#binaryData()}, {@link Audio#base64Data()}, or an
     * authorized {@link Audio#url()} under the Zero-Trust sandbox policy.
     */
    static byte[] loadAudioBytes(
        Audio audio,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        String contentKind
    ) {
        if (audio == null) {
            throw new IllegalArgumentException(contentKind + " contains null Audio");
        }
        byte[] binary = audio.binaryData();
        if (binary != null) {
            if (binary.length == 0) {
                throw new IllegalArgumentException(contentKind + " binaryData cannot be empty (0 bytes)");
            }
            if (binary.length > maxMediaBytes) {
                throw new IllegalArgumentException(
                    contentKind + " binaryData (" + binary.length +
                    " bytes) exceeds maxMediaBytes limit (" + maxMediaBytes + " bytes)"
                );
            }
            return binary;
        }
        String base64 = audio.base64Data();
        if (base64 != null && !base64.isEmpty()) {
            return decodeBoundedBase64(base64, maxMediaBytes, contentKind);
        }
        URI uri = audio.url();
        if (uri != null) {
            Path realFilePath = resolveAuthorizedLocalFile(
                uri, allowedMediaRoot, allowLocalFileMedia, maxMediaBytes, contentKind
            );
            try {
                byte[] fileBytes = Files.readAllBytes(realFilePath);
                if (fileBytes.length == 0 || fileBytes.length > maxMediaBytes) {
                    throw new IllegalArgumentException(
                        contentKind + " file size (" + fileBytes.length + " bytes) must be in [1, " + maxMediaBytes + "]"
                    );
                }
                return fileBytes;
            } catch (IOException e) {
                throw new IllegalArgumentException("Failed to read local audio file: " + realFilePath, e);
            }
        }
        throw new IllegalArgumentException(
            contentKind + " must specify non-empty binaryData, base64Data, or a valid file:// URI"
        );
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

    private static ArgusBitmap loadBitmapItem(
        ArgusMultimodalContext mctx,
        byte[] binaryData,
        String base64Data,
        URI uri,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        String contentKind
    ) {
        if (binaryData != null) {
            if (binaryData.length == 0) {
                throw new IllegalArgumentException(contentKind + " binaryData cannot be empty (0 bytes)");
            }
            if (binaryData.length > maxMediaBytes) {
                throw new IllegalArgumentException(
                    contentKind + " binaryData (" + binaryData.length +
                    " bytes) exceeds maxMediaBytes limit (" + maxMediaBytes + " bytes)"
                );
            }
            return loadBitmapFromBytes(mctx, binaryData, contentKind);
        }
        if (base64Data != null && !base64Data.isEmpty()) {
            byte[] decoded = decodeBoundedBase64(base64Data, maxMediaBytes, contentKind);
            return loadBitmapFromBytes(mctx, decoded, contentKind);
        }
        if (uri != null) {
            Path realFilePath = resolveAuthorizedLocalFile(
                uri, allowedMediaRoot, allowLocalFileMedia, maxMediaBytes, contentKind
            );
            ArgusBitmap bmp = ArgusBitmap.loadFile(mctx, realFilePath, false);
            if (bmp == null) {
                throw new ArgusNativeException(
                    ArgusNativeException.ERROR_INVALID_ARGUMENT,
                    "Failed to load " + contentKind + " file via ArgusBitmap.loadFile: " + realFilePath
                );
            }
            return bmp;
        }
        throw new IllegalArgumentException(
            contentKind + " must specify either non-empty base64Data or a valid file:// URI"
        );
    }

    private static ArgusBitmap loadBitmapFromBytes(
        ArgusMultimodalContext mctx,
        byte[] bytes,
        String contentKind
    ) {
        ArgusBitmap bmp = ArgusBitmap.loadBuffer(mctx, bytes, false);
        if (bmp == null) {
            throw new ArgusNativeException(
                ArgusNativeException.ERROR_INVALID_ARGUMENT,
                "Failed to decode " + contentKind + " buffer via ArgusBitmap.loadBuffer"
            );
        }
        return bmp;
    }

    @SuppressWarnings("deprecation") // Intentional: VideoItem.bitmap() retains independent ownership per frame in bitmaps list
    private static String readVideoFramesInto(
        ArgusMultimodalContext mctx,
        VideoContent vc,
        Path allowedMediaRoot,
        boolean allowLocalFileMedia,
        long maxMediaBytes,
        float videoFps,
        long videoTimestampIntervalMs,
        int maxVideoFrames,
        List<ArgusBitmap> outBitmaps
    ) {
        Video video = vc.video();
        if (video == null) {
            throw new IllegalArgumentException("VideoContent contains null Video");
        }
        if (!Float.isFinite(videoFps) || videoFps <= 0.0f) {
            throw new IllegalArgumentException("videoFps must be finite and > 0 (got " + videoFps + ")");
        }
        if (videoTimestampIntervalMs < 0L) {
            throw new IllegalArgumentException("videoTimestampIntervalMs must be >= 0 (got " + videoTimestampIntervalMs + ")");
        }
        if (maxVideoFrames <= 0) {
            throw new IllegalArgumentException("maxVideoFrames must be > 0 (got " + maxVideoFrames + ")");
        }

        final byte[] decodedBase64;
        final Path authorizedPath;
        if (video.base64Data() != null && !video.base64Data().isEmpty()) {
            decodedBase64 = decodeBoundedBase64(video.base64Data(), maxMediaBytes, "VideoContent");
            authorizedPath = null;
        } else if (video.url() != null) {
            decodedBase64 = null;
            authorizedPath = resolveAuthorizedLocalFile(
                video.url(), allowedMediaRoot, allowLocalFileMedia, maxMediaBytes, "VideoContent"
            );
        } else {
            throw new IllegalArgumentException(
                "VideoContent must specify either non-empty base64Data or a valid file:// URI"
            );
        }

        StringBuilder slotText = new StringBuilder();
        int framesExtracted = 0;
        try (ArgusVideo av = decodedBase64 != null
                ? ArgusVideo.loadBuffer(mctx, decodedBase64, videoFps, videoTimestampIntervalMs)
                : ArgusVideo.loadFile(mctx, authorizedPath, videoFps, videoTimestampIntervalMs)) {
            while (framesExtracted < maxVideoFrames) {
                ArgusVideo.VideoItem item = av.readNext();
                if (item == null) {
                    break;
                }
                if (item.text() != null && !item.text().isEmpty()) {
                    slotText.append(item.text().replace(
                        LibArgusChatFormatter.MEDIA_MARKER,
                        LibArgusChatFormatter.ESCAPED_MEDIA_MARKER
                    ));
                }
                if (item.bitmap() != null) {
                    outBitmaps.add(item.bitmap());
                    slotText.append(LibArgusChatFormatter.MEDIA_MARKER);
                    framesExtracted++;
                }
            }
        }
        if (framesExtracted == 0) {
            throw new IllegalArgumentException("VideoContent yielded 0 decodable video frames");
        }
        return slotText.toString();
    }

    static byte[] decodeBoundedBase64(String rawBase64, long maxMediaBytes, String contentKind) {
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
        return decoded;
    }

    static Path resolveAuthorizedLocalFile(
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
        return realFilePath;
    }
}
