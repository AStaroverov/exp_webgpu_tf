import type { StateBindings } from '../../../ppo/src/core/StateBindings.ts';
import { prepareRandomInputArrays, type InputArrays } from './InputArrays.ts';
import { createInputTensors } from './InputTensors.ts';

export const unknownStateBindings: StateBindings<InputArrays> = {
    createInputTensors,
    prepareRandomInputArrays,
};
