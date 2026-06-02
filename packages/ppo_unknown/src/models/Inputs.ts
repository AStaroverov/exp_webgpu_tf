/**
 * Network inputs for ppo_unknown — a single spatial board head
 * [ROWS, COLS, CHANNELS]. Mirrors tanks' `Inputs.ts` (createInputs) but with one
 * input instead of many groups. The returned object's value order must match the
 * tensor order produced by `state/InputTensors.ts`.
 */

import * as tf from '@tensorflow/tfjs';
import { BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS } from './dims.ts';

export function createInputs(name: string) {
    const boardInput = tf.input({
        name: name + '_boardInput',
        shape: [BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS],
    });

    return { boardInput };
}
