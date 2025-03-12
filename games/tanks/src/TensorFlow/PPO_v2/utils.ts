export type Actions = Float32Array;

export function readAction(action: Actions) {
    return {
        shoot: action[0] > 0,
        move: action[1],
        rotate: action[2],
        aim: new Float32Array([action[3], action[4]]),
    };
}