import { createBattlefield } from './createBattlefield.ts';
import { TankAgent } from './Agents/CurrentActorAgent.ts';
import { EntityId } from 'bitecs';

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;
    getAliveAgents(): TankAgent[];
    getAliveActors(): TankAgent[];
    getSuccessRatio(): number;

    addAgent(tankEid: EntityId, agent: TankAgent): void;
    getAgent(tankEid: EntityId): TankAgent | undefined;
    getAgents(): TankAgent[];
    getFreeTankEids(): EntityId[];
}