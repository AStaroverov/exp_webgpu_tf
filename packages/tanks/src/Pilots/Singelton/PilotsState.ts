export const PilotsState = {
    enabled: false,
    toggle: (on?: boolean) => {
        PilotsState.enabled = on ?? !PilotsState.enabled;
    },
};