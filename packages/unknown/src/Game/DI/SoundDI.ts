export const SoundDI: {
    enabled: boolean;
    destroy?: () => void;
    soundFrame?: (delta: number) => void;
} = {
    enabled: false,
    destroy: undefined,
    soundFrame: undefined,
};
