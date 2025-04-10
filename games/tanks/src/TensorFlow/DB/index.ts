import { Batch } from '../Common/Memory.ts';
import { createChannel } from '../../../../../lib/channles.ts';
import { Model } from '../Models/Transfer.ts';

export const memoryChannel = createChannel<{ memories: Batch, version: Record<Model, number> }>('memory-channel');

export const learnerStateChannel = createChannel<{
    model: Model,
    version: number,
    training: boolean,
}>('learner-state-channel');
