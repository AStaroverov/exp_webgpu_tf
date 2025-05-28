import { createBattlefield } from './createBattlefield.ts';
import { TankAgent } from './Agents/CurrentActorAgent.ts';
import { EntityId } from 'bitecs';

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