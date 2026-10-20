package cc.projectargus.libargus.internal;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ArgusUtf8StreamDecoderTest {

    @Test
    void testSplitMultiByteSequencesAcrossTokens() {
        // String containing ASCII, 2-byte (é), 3-byte (語), and 4-byte emoji (🔥 = F0 9F 94 A5, 🚀 = F0 9F 9A 80)
        String expected = "Hello café 日本語 🔥🚀 world!";
        byte[] raw = expected.getBytes(StandardCharsets.UTF_8);

        // 1. Feed 1 byte at a time (worst-case BPE byte-fallback splitting every multi-byte code point)
        ArgusUtf8StreamDecoder byteByByte = new ArgusUtf8StreamDecoder();
        List<String> pieces1 = new ArrayList<>();
        for (int i = 0; i < raw.length; i++) {
            byteByByte.accept(raw, i, 1, pieces1::add);
        }
        String result1 = byteByByte.finish(pieces1::add);
        assertEquals(expected, result1);
        assertEquals(expected, String.join("", pieces1));
        assertFalse(result1.contains("\uFFFD"), "Split valid UTF-8 must never produce U+FFFD replacement characters");

        // 2. Feed 2 bytes at a time
        ArgusUtf8StreamDecoder twoByTwo = new ArgusUtf8StreamDecoder();
        List<String> pieces2 = new ArrayList<>();
        for (int i = 0; i < raw.length; i += 2) {
            int len = Math.min(2, raw.length - i);
            twoByTwo.accept(raw, i, len, pieces2::add);
        }
        String result2 = twoByTwo.finish(pieces2::add);
        assertEquals(expected, result2);
        assertEquals(expected, String.join("", pieces2));

        // 3. Feed 3 bytes at a time
        ArgusUtf8StreamDecoder threeByThree = new ArgusUtf8StreamDecoder();
        List<String> pieces3 = new ArrayList<>();
        for (int i = 0; i < raw.length; i += 3) {
            int len = Math.min(3, raw.length - i);
            threeByThree.accept(raw, i, len, pieces3::add);
        }
        String result3 = threeByThree.finish(pieces3::add);
        assertEquals(expected, result3);
        assertEquals(expected, String.join("", pieces3));
    }

    @Test
    void testIncompleteTrailingSequenceReplacedOnFinish() {
        ArgusUtf8StreamDecoder decoder = new ArgusUtf8StreamDecoder();
        List<String> pieces = new ArrayList<>();
        // "A" followed by first 2 bytes of 4-byte emoji 🔥 (F0 9F)
        byte[] truncated = new byte[]{'A', (byte) 0xF0, (byte) 0x9F};
        decoder.accept(truncated, pieces::add);
        // Only "A" should be emitted before finish()
        assertEquals(List.of("A"), pieces);

        // finish() flushes the incomplete trailing sequence as U+FFFD
        String full = decoder.finish(pieces::add);
        assertEquals("A\uFFFD", full);
        assertEquals("A\uFFFD", String.join("", pieces));
    }

    @Test
    void testLargePieceBufferExpansion() {
        ArgusUtf8StreamDecoder decoder = new ArgusUtf8StreamDecoder();
        String large = "🔥αβγ".repeat(300); // > 256 bytes and > 256 chars
        byte[] bytes = large.getBytes(StandardCharsets.UTF_8);
        StringBuilder streamed = new StringBuilder();
        decoder.accept(bytes, streamed::append);
        String finished = decoder.finish(streamed::append);
        assertEquals(large, finished);
        assertEquals(large, streamed.toString());
    }
}
