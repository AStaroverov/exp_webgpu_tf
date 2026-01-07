export const PilotsState = {
    enabled: true,
    toggle: (on?: boolean) => {
        PilotsState.enabled = on ?? !PilotsState.enabled;
    },
};