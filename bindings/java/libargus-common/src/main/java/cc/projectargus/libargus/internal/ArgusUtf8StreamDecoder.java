package cc.projectargus.libargus.internal;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Stateful, RFC 3629-compliant incremental UTF-8 stream decoder for BPE token streaming.
 *
 * BPE tokenizers (e.g. Llama 3, Qwen, DeepSeek, Gemma) frequently split multi-byte UTF-8
 * code points (such as 3-byte CJK characters or 4-byte emojis) across consecutive token IDs.
 * Naive per-token {@code new String(bytes, UTF_8)} corrupts split code points into {@code U+FFFD}.
 *
 * This decoder buffers incomplete trailing UTF-8 lead/continuation bytes (up to 3 bytes)
 * across token boundaries and emits decoded {@link String} fragments only as complete Unicode
 * code points become available.
 */
public final class ArgusUtf8StreamDecoder {
    private final CharsetDecoder decoder;
    private ByteBuffer inBuf;
    private CharBuffer outBuf;
    private final StringBuilder fullText;

    public ArgusUtf8StreamDecoder() {
        this(true);
    }

    public ArgusUtf8StreamDecoder(boolean accumulateFullText) {
        this.decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE);
        this.inBuf = ByteBuffer.allocate(256);
        this.outBuf = CharBuffer.allocate(256);
        this.fullText = accumulateFullText ? new StringBuilder() : null;
    }

    /**
     * Feeds a raw token byte slice into the UTF-8 state machine and emits any completed
     * character sequences to {@code onPiece} (if non-null).
     *
     * @param bytes raw UTF-8 token piece bytes
     * @param offset starting index in {@code bytes}
     * @param length number of bytes to process
     * @param onPiece optional callback invoked when >= 1 chars are decoded
     */
    public void accept(byte[] bytes, int offset, int length, Consumer<String> onPiece) {
        Objects.requireNonNull(bytes, "bytes");
        if (offset < 0 || length < 0 || (long) offset + (long) length > bytes.length) {
            throw new IndexOutOfBoundsException("Invalid slice [offset=" + offset + ", length=" + length + "] for array of length " + bytes.length);
        }
        if (length == 0) {
            return;
        }
        ensureInputCapacity(length);
        inBuf.put(bytes, offset, length);
        inBuf.flip();
        decodeLoop(false, onPiece);
        inBuf.compact();
    }

    /**
     * Convenience overload feeding an entire {@code byte[]} array.
     */
    public void accept(byte[] bytes, Consumer<String> onPiece) {
        if (bytes != null && bytes.length > 0) {
            accept(bytes, 0, bytes.length, onPiece);
        }
    }

    /**
     * Feeds a raw token byte slice into the UTF-8 state machine and returns any newly completed
     * character sequences as a String (or {@code ""} if all bytes are buffered awaiting continuation).
     */
    public String decode(byte[] bytes, int offset, int length) {
        if (bytes == null || length == 0) {
            return "";
        }
        StringBuilder chunk = new StringBuilder();
        accept(bytes, offset, length, chunk::append);
        return chunk.toString();
    }

    /**
     * Flushes any incomplete trailing bytes at the end of a generation stream (replacing
     * any dangling incomplete UTF-8 sequence with {@code U+FFFD}) and resets the decoder.
     *
     * @param onPiece optional callback invoked if flushing produces characters
     * @return the complete accumulated text if accumulation was enabled, or {@code ""}
     */
    public String finish(Consumer<String> onPiece) {
        inBuf.flip();
        decodeLoop(true, onPiece);
        outBuf.clear();
        CoderResult cr = decoder.flush(outBuf);
        if (outBuf.position() > 0) {
            emitOutBuf(onPiece);
        }
        while (cr.isOverflow()) {
            growOutputBuffer();
            outBuf.clear();
            cr = decoder.flush(outBuf);
            if (outBuf.position() > 0) {
                emitOutBuf(onPiece);
            }
        }
        inBuf.clear();
        decoder.reset();
        return fullText != null ? fullText.toString() : "";
    }

    /**
     * Convenience overload for {@link #finish(Consumer)} with no piece callback.
     */
    public String finish() {
        return finish(null);
    }

    /**
     * Resets the decoder and clears any accumulated text.
     */
    public void reset() {
        inBuf.clear();
        outBuf.clear();
        decoder.reset();
        if (fullText != null) {
            fullText.setLength(0);
        }
    }

    private void ensureInputCapacity(int additionalBytes) {
        if (inBuf.remaining() < additionalBytes) {
            int unread = inBuf.position();
            int newCap = Math.max(inBuf.capacity() * 2, unread + additionalBytes);
            ByteBuffer next = ByteBuffer.allocate(newCap);
            inBuf.flip();
            next.put(inBuf);
            inBuf = next;
        }
    }

    private void growOutputBuffer() {
        outBuf = CharBuffer.allocate(outBuf.capacity() * 2);
    }

    private void decodeLoop(boolean endOfInput, Consumer<String> onPiece) {
        StringBuilder pieceBuilder = null;
        while (true) {
            outBuf.clear();
            CoderResult cr = decoder.decode(inBuf, outBuf, endOfInput);
            if (outBuf.position() > 0) {
                outBuf.flip();
                if (fullText != null) {
                    fullText.append(outBuf);
                    outBuf.rewind();
                }
                if (onPiece != null) {
                    if (cr.isOverflow()) {
                        if (pieceBuilder == null) {
                            pieceBuilder = new StringBuilder();
                        }
                        pieceBuilder.append(outBuf);
                    } else if (pieceBuilder != null) {
                        pieceBuilder.append(outBuf);
                        onPiece.accept(pieceBuilder.toString());
                        pieceBuilder = null;
                    } else {
                        onPiece.accept(outBuf.toString());
                    }
                }
            }
            if (cr.isOverflow()) {
                growOutputBuffer();
                continue;
            }
            break;
        }
        if (pieceBuilder != null && pieceBuilder.length() > 0 && onPiece != null) {
            onPiece.accept(pieceBuilder.toString());
        }
    }

    private void emitOutBuf(Consumer<String> onPiece) {
        outBuf.flip();
        if (fullText != null) {
            fullText.append(outBuf);
            outBuf.rewind();
        }
        if (onPiece != null) {
            onPiece.accept(outBuf.toString());
        }
    }
}
