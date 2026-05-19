import { EntityId } from 'bitecs';
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
    train: boolean;

    getVehicleEids(): readonly EntityId[];
    getTeamsCount(): number;
    getSuccessRatio(): number;
}