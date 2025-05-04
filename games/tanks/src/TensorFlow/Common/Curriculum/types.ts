import { createBattlefield } from './createBattlefield.ts';
import { TankAgent } from './Agents/CurrentActorAgent.ts';
import { EntityId } from 'bitecs';

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;
    getAgents: () => TankAgent[];
    getSuccessRatio: () => number;

    getActors(): TankAgent[];

    // private
    addAgent(tankEid: EntityId, agent: TankAgent): void;
    getAgent(tankEid: EntityId): TankAgent | undefined;
    getFreeTankEids(): EntityId[];
}