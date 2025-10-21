import { createChannel } from '../../../../lib/channles.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';
import { LearnData } from './Learner/createLearnerManager.ts';

import { CurriculumState } from '../../../ml-common/Curriculum/types.ts';
import { Model } from '../Models/def.ts';

export type AgentSample = {
    memoryBatch: AgentMemoryBatch,
    networkVersion: number,
}

export type EpisodeSample = {
    maxNetworkVersion: number,
    scenarioIndex: number,
    successRatio: number,
    isReference: boolean,
}

export const agentSampleChannel = createChannel<AgentSample>('agentSampleChannel');

export const episodeSampleChannel = createChannel<EpisodeSample>('episodeSampleChannel');

export const learnProcessChannel = createChannel<
    LearnData,
    { modelName: Model, version: number } | { modelName: Model, error: string, restart: boolean }
>('learn-memory-channel');

export const queueSizeChannel = createChannel<number>('queueSizeChannel');

export const modelSettingsChannel = createChannel<{ lr?: number, perturbChance?: number, perturbScale?: number, expIteration?: number }>('modelSettingsChannel');

export const curriculumStateChannel = createChannel<CurriculumState>('curriculumStateChannel');