import { createChannel } from "../../../../lib/channles.ts";
import { AgentMemoryBatch } from "../memory/Memory.ts";
import { LearnData } from "../learner/createLearnerManager.ts";
import { Model } from "../models/def.ts";

// AgentSample/LearnData carry their state type opaquely; the generic
// learner/actor factories re-narrow to <S> at the producer/consumer boundary.
export type AgentSample = {
  memoryBatch: AgentMemoryBatch<unknown>;
  networkVersion: number;
  scenarioIndex: number;
};

export type EpisodeSample = {
  maxNetworkVersion: number;
  scenarioIndex: number;
  successRatio: number;
  isReference: boolean;
};

export const agentSampleChannel = createChannel<AgentSample>("agentSampleChannel");

export const episodeSampleChannel = createChannel<EpisodeSample>("episodeSampleChannel");

export const learnProcessChannel = createChannel<
  LearnData<unknown>,
  { modelName: Model; version: number } | { modelName: Model; error: string; restart: boolean }
>("learn-memory-channel");

export const queueSizeChannel = createChannel<number>("queueSizeChannel");

export type { ModelSettings } from "./types.ts";
import type { ModelSettings } from "./types.ts";
export const modelSettingsChannel = createChannel<ModelSettings>("modelSettingsChannel");
