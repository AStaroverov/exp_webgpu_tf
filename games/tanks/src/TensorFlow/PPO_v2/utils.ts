export type Actions = Float32Array;

export function createAction(shoot: number, move: Float32Array, aim: Float32Array): Actions {
    return new Float32Array([shoot, move[0], move[1], aim[0], aim[1]]);
}

export function readAction(action: Actions) {
    return {
        shoot: action[0] > 0.5,
        move: action[1],
        rotate: action[2],
        aim: new Float32Array([action[3], action[4]]),
    };
}