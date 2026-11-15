package cc.projectargus.libargus.langchain4j;

import java.util.function.BiConsumer;

/**
 * Stateful, BPE-boundary-safe finite-state transducer separating {@code <think>...</think>}
 * reasoning traces from visible assistant text across arbitrary multi-piece token boundaries.
 *
 * <p>Guarantees that partial tag prefixes (e.g. {@code "<thi"} or {@code "</thi"}) are held back
 * until confirmed or refuted, preventing tag fragments from ever leaking into streaming callbacks,
 * while flushing any unclosed reasoning suffix cleanly upon stream completion.
 */
final class LibArgusThinkingParser {

    static final String OPEN_TAG = "<think>";
    static final String CLOSE_TAG = "</think>";
    private static final int MAX_LEADING_WHITESPACE = 32;

    private static final int STATE_DETECT_OPEN = 0;
    private static final int STATE_STRIP_POST_OPEN_NEWLINE = 1;
    private static final int STATE_IN_THINKING = 2;
    private static final int STATE_STRIP_POST_CLOSE_NEWLINE = 3;
    private static final int STATE_IN_RESPONSE = 4;

    record ParsedOutput(String text, String thinking) {}

    private final BiConsumer<String, Boolean> chunkSink;
    private final StringBuilder pending = new StringBuilder(32);
    private final StringBuilder fullThinking = new StringBuilder();
    private final StringBuilder fullResponse = new StringBuilder();
    private int state;
    private boolean sawThinking;
    private boolean sawCarriageReturnAfterOpen;

    /**
     * @param promptEndsWithOpenTag true if the formatted prompt already ends with {@code <think>}
     * @param chunkSink             optional streaming callback receiving {@code (chunk, isThinking)}
     */
    LibArgusThinkingParser(boolean promptEndsWithOpenTag, BiConsumer<String, Boolean> chunkSink) {
        this.chunkSink = chunkSink;
        if (promptEndsWithOpenTag) {
            this.state = STATE_STRIP_POST_OPEN_NEWLINE;
            this.sawThinking = true;
        } else {
            this.state = STATE_DETECT_OPEN;
            this.sawThinking = false;
        }
    }

    static boolean promptEndsWithOpenThinkTag(String formattedPrompt) {
        if (formattedPrompt == null || formattedPrompt.isEmpty()) {
            return false;
        }
        int end = formattedPrompt.length();
        while (end > 0 && Character.isWhitespace(formattedPrompt.charAt(end - 1))) {
            end--;
        }
        return end >= OPEN_TAG.length()
            && formattedPrompt.regionMatches(end - OPEN_TAG.length(), OPEN_TAG, 0, OPEN_TAG.length());
    }

    static ParsedOutput parseComplete(String rawText, boolean promptEndsWithOpenTag) {
        LibArgusThinkingParser parser = new LibArgusThinkingParser(promptEndsWithOpenTag, null);
        if (rawText != null && !rawText.isEmpty()) {
            parser.accept(rawText);
        }
        return parser.finish();
    }

    void accept(String piece) {
        if (piece == null || piece.isEmpty()) {
            return;
        }
        if (state == STATE_IN_RESPONSE) {
            emitResponse(piece);
            return;
        }
        pending.append(piece);
        drainPending();
    }

    ParsedOutput finish() {
        if (pending.length() > 0) {
            if (state == STATE_DETECT_OPEN) {
                int wsLen = countLeadingWhitespace(pending);
                String afterWs = pending.substring(wsLen);
                if (OPEN_TAG.equals(afterWs)) {
                    sawThinking = true;
                } else {
                    emitResponse(pending.toString());
                }
            } else if (state == STATE_IN_THINKING) {
                emitThinking(pending.toString());
            } else if (state == STATE_IN_RESPONSE) {
                emitResponse(pending.toString());
            }
            pending.setLength(0);
        }
        String thinkingOut = sawThinking ? fullThinking.toString() : null;
        return new ParsedOutput(fullResponse.toString(), thinkingOut);
    }

    private void drainPending() {
        boolean progress = true;
        while (progress && pending.length() > 0) {
            progress = false;
            switch (state) {
                case STATE_DETECT_OPEN -> {
                    int wsLen = countLeadingWhitespace(pending);
                    if (wsLen == pending.length()) {
                        if (wsLen > MAX_LEADING_WHITESPACE) {
                            String flushed = pending.toString();
                            pending.setLength(0);
                            state = STATE_IN_RESPONSE;
                            emitResponse(flushed);
                        }
                        return;
                    }
                    int remLen = pending.length() - wsLen;
                    int cmpLen = Math.min(remLen, OPEN_TAG.length());
                    if (!regionMatches(pending, wsLen, OPEN_TAG, cmpLen)) {
                        String flushed = pending.toString();
                        pending.setLength(0);
                        state = STATE_IN_RESPONSE;
                        emitResponse(flushed);
                        return;
                    }
                    if (remLen >= OPEN_TAG.length()) {
                        pending.delete(0, wsLen + OPEN_TAG.length());
                        sawThinking = true;
                        state = STATE_STRIP_POST_OPEN_NEWLINE;
                        progress = true;
                    }
                }
                case STATE_STRIP_POST_OPEN_NEWLINE -> {
                    char c0 = pending.charAt(0);
                    if (!sawCarriageReturnAfterOpen && c0 == '\r') {
                        pending.deleteCharAt(0);
                        sawCarriageReturnAfterOpen = true;
                        progress = true;
                    } else if (c0 == '\n') {
                        pending.deleteCharAt(0);
                        state = STATE_IN_THINKING;
                        progress = true;
                    } else {
                        state = STATE_IN_THINKING;
                        progress = true;
                    }
                }
                case STATE_IN_THINKING -> {
                    int closeIdx = pending.indexOf(CLOSE_TAG);
                    if (closeIdx >= 0) {
                        if (closeIdx > 0) {
                            emitThinking(pending.substring(0, closeIdx));
                        }
                        pending.delete(0, closeIdx + CLOSE_TAG.length());
                        state = STATE_STRIP_POST_CLOSE_NEWLINE;
                        progress = true;
                    } else {
                        int holdback = longestSuffixMatchingPrefix(pending, CLOSE_TAG);
                        int safeLen = pending.length() - holdback;
                        if (safeLen > 0) {
                            emitThinking(pending.substring(0, safeLen));
                            pending.delete(0, safeLen);
                        }
                    }
                }
                case STATE_STRIP_POST_CLOSE_NEWLINE -> {
                    int strip = 0;
                    while (strip < pending.length()) {
                        char ch = pending.charAt(strip);
                        if (ch == '\r' || ch == '\n') {
                            strip++;
                        } else {
                            break;
                        }
                    }
                    if (strip > 0) {
                        pending.delete(0, strip);
                    }
                    if (pending.length() > 0) {
                        String rest = pending.toString();
                        pending.setLength(0);
                        state = STATE_IN_RESPONSE;
                        emitResponse(rest);
                    }
                }
                case STATE_IN_RESPONSE -> {
                    String rest = pending.toString();
                    pending.setLength(0);
                    emitResponse(rest);
                }
                default -> throw new IllegalStateException("Unknown parser state: " + state);
            }
        }
    }

    private void emitThinking(String chunk) {
        if (chunk.isEmpty()) {
            return;
        }
        fullThinking.append(chunk);
        if (chunkSink != null) {
            chunkSink.accept(chunk, Boolean.TRUE);
        }
    }

    private void emitResponse(String chunk) {
        if (chunk.isEmpty()) {
            return;
        }
        fullResponse.append(chunk);
        if (chunkSink != null) {
            chunkSink.accept(chunk, Boolean.FALSE);
        }
    }

    private static int countLeadingWhitespace(StringBuilder sb) {
        int i = 0;
        int len = sb.length();
        while (i < len && Character.isWhitespace(sb.charAt(i))) {
            i++;
        }
        return i;
    }

    private static boolean regionMatches(StringBuilder sb, int sbOffset, String target, int length) {
        for (int i = 0; i < length; i++) {
            if (sb.charAt(sbOffset + i) != target.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static int longestSuffixMatchingPrefix(StringBuilder sb, String tag) {
        int maxCheck = Math.min(sb.length(), tag.length() - 1);
        for (int len = maxCheck; len >= 1; len--) {
            if (regionMatches(sb, sb.length() - len, tag, len)) {
                return len;
            }
        }
        return 0;
    }
}
