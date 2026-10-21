package cc.projectargus.libargus.internal;

import cc.projectargus.libargus.ArgusNativeException;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * Thread-safe sequence slot lease manager for zero-rollback sequence forking ({@code forkSlot} / {@code freeSlot}),
 * shared across {@code :libargus-core} (Panama FFM) and {@code :libargus-jni} (JNI).
 */
public final class ArgusSequenceSlotPool {

    @FunctionalInterface
    public interface SlotCopier {
        boolean copy(int srcSeqId, int dstSeqId);
    }

    private final BitSet leasedSlots = new BitSet();
    private final BitSet releasingSlots = new BitSet();
    private final Object slotPoolLock = new Object();
    private final int cloneSlotStart;
    private final int cloneSlotCount;

    public ArgusSequenceSlotPool(int maxSlots, int configuredCloneSlots) {
        int safeMax = Math.max(0, maxSlots);
        int clampedClone = Math.max(0, Math.min(configuredCloneSlots, safeMax));
        this.cloneSlotCount = clampedClone;
        this.cloneSlotStart = Math.max(0, safeMax - clampedClone);
        this.leasedSlots.set(0); // Slot 0 is reserved as root sequence slot
    }

    public int forkSlot(int srcSeqId, int maxSlots, SlotCopier copier) {
        ArgusCommonValidation.checkNonNegative(srcSeqId, "srcSeqId");
        Objects.requireNonNull(copier, "copier");
        if (srcSeqId >= maxSlots) {
            throw new IllegalArgumentException("srcSeqId " + srcSeqId + " exceeds max sequence slots " + maxSlots);
        }
        if (cloneSlotCount == 0) {
            throw new IllegalStateException("Clone pool unconfigured (cloneSlots = 0); increase cloneSlots in ArgusContextConfig");
        }
        int leasedSlot;
        synchronized (slotPoolLock) {
            leasedSlot = -1;
            for (int i = cloneSlotStart; i < cloneSlotStart + cloneSlotCount; i++) {
                if (!leasedSlots.get(i) && !releasingSlots.get(i)) {
                    leasedSlot = i;
                    leasedSlots.set(i);
                    break;
                }
            }
        }
        if (leasedSlot < 0) {
            throw new IllegalStateException("All " + cloneSlotCount + " clone worker slots are currently leased; increase cloneSlots in ArgusContextConfig");
        }
        boolean success = false;
        try {
            boolean ok = copier.copy(srcSeqId, leasedSlot);
            if (!ok) {
                ArgusNativeException.throwLastError("forkSlot");
            }
            success = true;
            return leasedSlot;
        } finally {
            if (!success) {
                synchronized (slotPoolLock) {
                    leasedSlots.clear(leasedSlot);
                }
            }
        }
    }

    public void freeSlot(int seqId, IntPredicate slotCleaner) {
        ArgusCommonValidation.checkNonNegative(seqId, "seqId");
        Objects.requireNonNull(slotCleaner, "slotCleaner");
        if (cloneSlotCount == 0 || seqId < cloneSlotStart || seqId >= cloneSlotStart + cloneSlotCount) {
            throw new IllegalArgumentException(
                "seqId " + seqId + " does not belong to the configured clone pool ["
                    + cloneSlotStart + ", " + (cloneSlotStart + cloneSlotCount) + ")"
            );
        }
        synchronized (slotPoolLock) {
            if (!leasedSlots.get(seqId) || releasingSlots.get(seqId)) {
                throw new IllegalStateException("Slot " + seqId + " is not currently leased or is already being released");
            }
            releasingSlots.set(seqId);
        }
        boolean ok = false;
        try {
            ok = slotCleaner.test(seqId);
            if (!ok) {
                ArgusNativeException.throwLastError("freeSlot");
            }
        } finally {
            synchronized (slotPoolLock) {
                if (ok) {
                    leasedSlots.clear(seqId);
                    releasingSlots.clear(seqId);
                } else {
                    releasingSlots.clear(seqId);
                }
            }
        }
    }

    public int getAvailableSlotCount() {
        if (cloneSlotCount == 0) {
            return 0;
        }
        synchronized (slotPoolLock) {
            int inUse = 0;
            for (int i = cloneSlotStart; i < cloneSlotStart + cloneSlotCount; i++) {
                if (leasedSlots.get(i) || releasingSlots.get(i)) {
                    inUse++;
                }
            }
            return Math.max(0, cloneSlotCount - inUse);
        }
    }
}
