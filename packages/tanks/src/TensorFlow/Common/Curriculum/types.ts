import { createBattlefield } from './createBattlefield.ts';
import { EntityId } from 'bitecs';
import { TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { ValueOf } from '../../../../../../lib/Types';
import { PilotType } from '../../../Pilots/Components/Pilot.ts';

export type Scenario = Awaited<ReturnType<typeof createBattlefield>> & {
    index: number;

    getTankEids(): readonly EntityId[];
    getTeamsCount(): number;

    getAlivePilots(): readonly TankAgent[];
    getAliveActors(): readonly TankAgent[];
    getSuccessRatio(): number;

    setPilot(tankEid: EntityId, type: ValueOf<typeof PilotType>): void;
    getPilot(tankEid: EntityId): TankAgent | undefined;
    getPilots(): readonly TankAgent[];
    getFreeTankEids(): readonly EntityId[];
}