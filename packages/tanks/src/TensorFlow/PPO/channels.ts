import { AgentMemoryBatch } from '../Common/Memory.ts';
import { createChannel } from '../../../../../lib/channles.ts';
import { LearnData } from './Learner/createLearnerManager.ts';

import { Model } from '../Models/def.ts';

export type EpisodeSample = {
    memoryBatch: AgentMemoryBatch,
    networkVersion: number,
    scenarioIndex: number,
    successRatio: number,
}

export const episodeSampleChannel = createChannel<EpisodeSample>('episodeSampleChannel');

export const learnProcessChannel = createChannel<
    LearnData,
    { modelName: Model, version: number } | { modelName: Model, error: string }
>('learn-memory-channel');

export const queueSizeChannel = createChannel<number>('queueSizeChannel');

export const learningRateChannel = createChannel<number>('learningRateChannel');

export type CurriculumState = {
    mapScenarioIndexToSuccessRatio: Record<number, number>,
}

export const curriculumStateChannel = createChannel<CurriculumState>('curriculumStateChannel');