export type Actions = Float32Array;

export function readActions(actions: Actions) {
    return {
        shoot: (actions[0]) > 0.5,
        move: (actions[1]),
        rotate: (actions[2]),
        aimX: (actions[3]),
        aimY: (actions[4]),
    };
}
