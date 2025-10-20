import { EntityId } from 'bitecs';
import { TankAgent } from '../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { createBattlefield } from './createBattlefield.ts';

export type CurriculumState = {
    currentVersion: number,
    mapScenarioIndexToSuccessRatio: Record<number, number>,
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
    currentVersion: 0,
    mapScenarioIndexToSuccessRatio: {},
};

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;

    getTankEids(): readonly EntityId[];
    getTeamsCount(): number;

    getAlivePilots(): readonly TankAgent[];
    getAliveActors(): readonly TankAgent[];
    getSuccessRatio(): number;

    setPilot(tankEid: EntityId, agent: TankAgent): void;
    getPilot(tankEid: EntityId): TankAgent | undefined;
    getPilots(): readonly TankAgent[];
    getFreeTankEids(): readonly EntityId[];
}