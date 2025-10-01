import { createChannel } from '../../../../lib/channles.ts';
import { Model } from '../Models/def.ts';
import type { LearnData } from './Learner/createLearnerManager.ts';

export const learnProcessChannel = createChannel<
    LearnData,
    { modelName: Model, version: number } | { modelName: Model, error: string }
>('learn-process-channel');

export const learningRateChannel = createChannel<number>('learning-rate-channel');
