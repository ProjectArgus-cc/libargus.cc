package cc.projectargus.libargus.internal;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Predicate;

/**
 * Stateful, BPE-boundary-safe stop-sequence detector with sliding prefix holdback.
 *
 * <p>Because BPE tokenizers frequently split multi-character stop sequences (such as {@code "<|im_end|>"}
 * or {@code "Observation:"}) across multiple consecutive tokens, naive per-token checks either miss
 * split stop sequences or leak leading characters of a stop sequence to streaming callbacks before
 * the full stop sequence is recognized.
 *
 * <p>This detector holds back only the trailing suffix of decoded text that is a proper prefix of
 * at least one configured stop sequence, emitting all unambiguous characters immediately and stripping
 * any matched stop sequence from both the streamed output and the final accumulated string.
 */
public final class ArgusStopSequenceDetector {
    private static final String[] EMPTY_STOPS = new String[0];

    private final String[] stopSequences;
    private final StringBuilder emittedText = new StringBuilder();
    private final StringBuilder pending = new StringBuilder();
    private String matchedStopSequence;
    private boolean stoppedByCallback;

    public ArgusStopSequenceDetector(Collection<String> stopSequences) {
        if (stopSequences == null || stopSequences.isEmpty()) {
            this.stopSequences = EMPTY_STOPS;
        } else {
            List<String> filtered = new ArrayList<>(stopSequences.size());
            for (String s : stopSequences) {
                if (s != null && !s.isEmpty()) {
                    filtered.add(s);
                }
            }
            this.stopSequences = filtered.isEmpty() ? EMPTY_STOPS : filtered.toArray(EMPTY_STOPS);
        }
    }

    /**
     * Processes a newly decoded UTF-8 piece, emitting any unambiguous prefix to {@code shouldContinue}.
     *
     * @param piece          decoded UTF-8 character sequence from the current token
     * @param shouldContinue optional streaming callback predicate; returning {@code false} halts generation
     * @return {@code true} if generation should continue, {@code false} if a stop sequence matched or callback stopped
     */
    public boolean acceptPiece(String piece, Predicate<String> shouldContinue) {
        if (piece == null || piece.isEmpty()) {
            return true;
        }

        if (stopSequences.length == 0) {
            emittedText.append(piece);
            if (shouldContinue != null && !shouldContinue.test(piece)) {
                stoppedByCallback = true;
                return false;
            }
            return true;
        }

        pending.append(piece);

        // 1. Check for the earliest complete stop-sequence match in pending.
        int bestIdx = -1;
        String bestStop = null;
        for (String stop : stopSequences) {
            int idx = pending.indexOf(stop);
            if (idx >= 0) {
                if (bestIdx == -1 || idx < bestIdx || (idx == bestIdx && stop.length() > bestStop.length())) {
                    bestIdx = idx;
                    bestStop = stop;
                }
            }
        }

        if (bestIdx >= 0) {
            matchedStopSequence = bestStop;
            String safePrefix = pending.substring(0, bestIdx);
            pending.setLength(0);
            if (!safePrefix.isEmpty()) {
                emittedText.append(safePrefix);
                if (shouldContinue != null && !shouldContinue.test(safePrefix)) {
                    stoppedByCallback = true;
                }
            }
            return false;
        }

        // 2. Determine the longest suffix of pending that is a proper prefix of any stop sequence.
        int pLen = pending.length();
        int holdbackStart = pLen;
        for (String stop : stopSequences) {
            int maxPrefixLen = Math.min(pLen, stop.length() - 1);
            for (int k = maxPrefixLen; k >= 1; k--) {
                if (regionMatches(pending, pLen - k, stop, 0, k)) {
                    int candidateStart = pLen - k;
                    if (candidateStart < holdbackStart) {
                        holdbackStart = candidateStart;
                    }
                    break;
                }
            }
        }

        if (holdbackStart > 0) {
            String safeChunk = pending.substring(0, holdbackStart);
            pending.delete(0, holdbackStart);
            emittedText.append(safeChunk);
            if (shouldContinue != null && !shouldContinue.test(safeChunk)) {
                stoppedByCallback = true;
                return false;
            }
        }

        return true;
    }

    /**
     * Flushes any held-back partial prefix when generation ends without matching a stop sequence,
     * and returns the complete emitted text.
     */
    public String finish(Predicate<String> shouldContinue) {
        if (matchedStopSequence == null && !stoppedByCallback && pending.length() > 0) {
            String remainder = pending.toString();
            pending.setLength(0);
            emittedText.append(remainder);
            if (shouldContinue != null && !shouldContinue.test(remainder)) {
                stoppedByCallback = true;
            }
        } else {
            pending.setLength(0);
        }
        return emittedText.toString();
    }

    public String matchedStopSequence() {
        return matchedStopSequence;
    }

    public boolean isStoppedByStopSequence() {
        return matchedStopSequence != null;
    }

    public boolean isStoppedByCallback() {
        return stoppedByCallback;
    }

    private static boolean regionMatches(CharSequence seq, int toffset, String other, int ooffset, int len) {
        for (int i = 0; i < len; i++) {
            if (seq.charAt(toffset + i) != other.charAt(ooffset + i)) {
                return false;
            }
        }
        return true;
    }
}
