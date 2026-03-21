// History frame offsets from current step (exponentially growing gaps)
// Frames: t, t-2, t-6, t-14
export const HISTORY_OFFSETS = [0, 2, 6, 14] as const;
export const HISTORY_LENGTH = HISTORY_OFFSETS.length;
