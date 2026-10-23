package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusGenerationResult;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ArgusStopSequenceDetectorTest {

    @Test
    public void testSplitTokenStopSequenceNeverLeaksToStream() {
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(List.of("<|im_end|>", "Observation:"));
        List<String> streamedPieces = new ArrayList<>();

        assertTrue(detector.acceptPiece("Hello ", streamedPieces::add));
        assertTrue(detector.acceptPiece("world", streamedPieces::add));
        // Split "<|im_end|>" across 3 tokens, with trailing garbage in the 3rd token:
        assertTrue(detector.acceptPiece("<|", streamedPieces::add));
        assertTrue(detector.acceptPiece("im_", streamedPieces::add));
        assertFalse(detector.acceptPiece("end|> trailing garbage", streamedPieces::add));

        String finalText = detector.finish(streamedPieces::add);
        assertEquals("Hello world", finalText);
        assertEquals("Hello world", String.join("", streamedPieces));
        assertTrue(detector.isStoppedByStopSequence());
        assertFalse(detector.isStoppedByCallback());
        assertEquals("<|im_end|>", detector.matchedStopSequence());
    }

    @Test
    public void testFalseAlarmPrefixIsFlushedImmediatelyUponDivergence() {
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(List.of("<|im_end|>"));
        List<String> streamedPieces = new ArrayList<>();

        assertTrue(detector.acceptPiece("Prefix ", streamedPieces::add));
        assertTrue(detector.acceptPiece("<|im_", streamedPieces::add));
        // At this point "<|im_" is held back
        assertEquals("Prefix ", String.join("", streamedPieces));

        // Next token diverges from "<|im_end|>":
        assertTrue(detector.acceptPiece("start|> continues", streamedPieces::add));
        assertEquals("Prefix <|im_start|> continues", String.join("", streamedPieces));

        String finalText = detector.finish(streamedPieces::add);
        assertEquals("Prefix <|im_start|> continues", finalText);
        assertFalse(detector.isStoppedByStopSequence());
        assertNull(detector.matchedStopSequence());
    }

    @Test
    public void testHeldBackTrailingPrefixFlushedOnFinishWhenNotMatched() {
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(List.of("STOP_NOW"));
        List<String> streamedPieces = new ArrayList<>();

        assertTrue(detector.acceptPiece("Complete output ", streamedPieces::add));
        assertTrue(detector.acceptPiece("STOP_", streamedPieces::add));
        assertEquals("Complete output ", String.join("", streamedPieces));

        // Generation hits EOG or maxTokens before "NOW" ever arrives:
        String finalText = detector.finish(streamedPieces::add);
        assertEquals("Complete output STOP_", finalText);
        assertEquals("Complete output STOP_", String.join("", streamedPieces));
        assertFalse(detector.isStoppedByStopSequence());
    }

    @Test
    public void testOverlappingStopSequencesAndSafePrefixFlush() {
        // "ABY" and "BCZ": "AB" is held back for "ABY"; when "C" arrives, "A" is safe and flushed while "BC" is held back for "BCZ"
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(List.of("ABY", "BCZ"));
        List<String> streamedPieces = new ArrayList<>();

        assertTrue(detector.acceptPiece("AB", streamedPieces::add));
        assertEquals("", String.join("", streamedPieces));

        assertTrue(detector.acceptPiece("C", streamedPieces::add));
        assertEquals("A", String.join("", streamedPieces));

        assertFalse(detector.acceptPiece("Z_after", streamedPieces::add));
        String finalText = detector.finish(streamedPieces::add);
        assertEquals("A", finalText);
        assertEquals("A", String.join("", streamedPieces));
        assertEquals("BCZ", detector.matchedStopSequence());
    }

    @Test
    public void testMultiByteUtf8EmojiSplitAndStopSequence() {
        ArgusUtf8StreamDecoder utf8 = new ArgusUtf8StreamDecoder(false);
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(List.of("🛑STOP"));
        StringBuilder streamed = new StringBuilder();

        byte[] rocket = "🚀".getBytes(StandardCharsets.UTF_8); // 4 bytes
        byte[] stopSign = "🛑".getBytes(StandardCharsets.UTF_8); // 4 bytes
        byte[] stopWord = "STOP!".getBytes(StandardCharsets.UTF_8);

        // Feed rocket split in half (2 bytes + 2 bytes)
        assertTrue(utf8.acceptUntil(rocket, 0, 2, p -> detector.acceptPiece(p, s -> { streamed.append(s); return true; })));
        assertTrue(utf8.acceptUntil(rocket, 2, 2, p -> detector.acceptPiece(p, s -> { streamed.append(s); return true; })));
        assertEquals("🚀", streamed.toString());

        // Feed stopSign split in half, followed by "STOP!"
        assertTrue(utf8.acceptUntil(stopSign, 0, 2, p -> detector.acceptPiece(p, s -> { streamed.append(s); return true; })));
        assertTrue(utf8.acceptUntil(stopSign, 2, 2, p -> detector.acceptPiece(p, s -> { streamed.append(s); return true; })));
        assertEquals("🚀", streamed.toString()); // "🛑" is held back!

        assertFalse(utf8.acceptUntil(stopWord, 0, stopWord.length, p -> detector.acceptPiece(p, s -> { streamed.append(s); return true; })));
        String finalText = detector.finish(s -> { streamed.append(s); return true; });
        assertEquals("🚀", finalText);
        assertEquals("🚀", streamed.toString());
        assertEquals("🛑STOP", detector.matchedStopSequence());
    }

    @Test
    public void testNullAndEmptyStopSequencesIgnoredAndCallbackHalt() {
        ArgusStopSequenceDetector detector = new ArgusStopSequenceDetector(Arrays.asList(null, "", "END"));
        List<String> streamed = new ArrayList<>();

        assertFalse(detector.acceptPiece("First chunk", piece -> {
            streamed.add(piece);
            return false; // Halt via callback predicate
        }));
        assertTrue(detector.isStoppedByCallback());
        assertFalse(detector.isStoppedByStopSequence());
        assertEquals("First chunk", detector.finish(streamed::add));
        assertEquals(List.of("First chunk"), streamed);

        ArgusGenerationResult result = new ArgusGenerationResult(
            "First chunk",
            ArgusGenerationResult.StopReason.STOPPED_BY_CALLBACK,
            4,
            1,
            null
        );
        assertEquals(ArgusGenerationResult.StopReason.STOPPED_BY_CALLBACK, result.stopReason());
        assertThrows(IllegalArgumentException.class, () ->
            new ArgusGenerationResult("x", ArgusGenerationResult.StopReason.EOG, -1, 0, null));
    }
}
