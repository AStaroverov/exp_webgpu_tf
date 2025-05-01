import { createBattlefield } from './createBattlefield.ts';
import { TankAgent } from '../../PPO/Actor/ActorAgent.ts';
import { EntityId } from 'bitecs';

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;
    getAgents: () => TankAgent[];
    getSuccessRatio: () => number;

    // private
    addAgent(tankEid: EntityId, agent: TankAgent): void;
    getAgent(tankEid: EntityId): TankAgent | undefined;
}