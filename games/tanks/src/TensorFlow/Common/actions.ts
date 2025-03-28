import { clamp } from 'lodash-es';

export type Actions = Float32Array;

export function readActions(actions: Actions) {
    return {
        shoot: clamp(actions[0], -1, 1) > 0,
        move: clamp(actions[1], -1, 1),
        rotate: clamp(actions[2], -1, 1),
        aimX: clamp(actions[3], -2, 2),
        aimY: clamp(actions[4], -2, 2),
    };
}
