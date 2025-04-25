import { Batch } from '../Common/Memory.ts';
import { createChannel } from '../../../../../lib/channles.ts';
import { LearnBatch } from './Learner/createLearnerManager.ts';
import { Model } from '../Models/Transfer.ts';

export const memoryChannel = createChannel<{ memories: Batch, version: number }>('memory-channel');

export const learnMemoryChannel = createChannel<
    LearnBatch,
    { modelName: Model, version: number } | { modelName: Model, error: string }
>('learn-memory-channel');

export const learningRateChannel = createChannel<number>('learning-rate-channel');

export const learnerStateChannel = createChannel<{
    version: number,
    training: boolean,
}>('learner-state-channel');
