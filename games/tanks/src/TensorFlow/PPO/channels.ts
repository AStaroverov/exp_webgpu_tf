import { Batch } from '../Common/Memory.ts';
import { createChannel } from '../../../../../lib/channles.ts';
import { LearnBatch } from './Learner/createLearnerManager.ts';
import { Model } from '../Models/Transfer.ts';

export const actorMemoryChannel = createChannel<{ memories: Batch, version: number }>('memory-channel');

export const learnMemoryChannel = createChannel<
    LearnBatch,
    { modelName: Model, version: number } | { modelName: Model, error: string }
>('learn-memory-channel');

export const learningRateChannel = createChannel<number>('learningRateChannel');

export const queueSizeChannel = createChannel<number>('queueSizeChannel');
