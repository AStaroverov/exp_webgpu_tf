import { EntityId } from 'bitecs';
import { TankAgent } from '../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { createBattlefield } from './createBattlefield.ts';

export type CurriculumState = {
    iteration: number,
    mapScenarioIndexToSuccessRatio: Record<number, number>,
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
    iteration: 0,
    mapScenarioIndexToSuccessRatio: {},
};

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;
    isTrain: boolean;

    getVehicleEids(): readonly EntityId[];
    getTeamsCount(): number;

    getAlivePilots(): readonly TankAgent[];
    getAliveActors(): readonly TankAgent[];
    getSuccessRatio(): number;

    setPilot(vehicleEid: EntityId, agent: TankAgent): void;
    getPilot(vehicleEid: EntityId): TankAgent | undefined;
    getPilots(): readonly TankAgent[];
    getFreeVehicleEids(): readonly EntityId[];
}