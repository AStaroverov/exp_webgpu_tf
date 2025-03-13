import { tanh } from '../../../../../lib/math.ts';

export type Actions = Float32Array;

export function readAction(action: Actions) {
    return {
        shoot: tanh(action[0]) > 0,
        move: tanh(action[1]),
        rotate: tanh(action[2]),
        aimX: tanh(action[3]) * 2,
        aimY: tanh(action[4]) * 2,
    };
}