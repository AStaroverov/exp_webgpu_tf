import { Batch } from '../Common/Memory.ts';
import { createChannel } from '../../../../../lib/channles.ts';

export const memoryChannel = createChannel<{ memories: Batch, version: number }>('memory-channel');

export const learnerStateChannel = createChannel<{
    version: number,
    training: boolean,
}>('learner-state-channel');
