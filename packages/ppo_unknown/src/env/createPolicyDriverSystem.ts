/**
 * createPolicyDriverSystem — the ML decision driver, installed as a
 * `SystemGroup.Before` plugin IN PLACE OF the game's `createStandInDriverSystem`.
 *
 * Per tick: find living agent-controlled tanks that present an open slot
 * (`needsDecision`), refresh the board observation once for all of them, then let
 * each agent decide → record → enqueue. Exactly the seam the stand-in used.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { needsDecision } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';
import { snapshotUnknownBoard } from '../state/snapshotUnknownBoard.ts';
import { scoreTracker } from '../reward/ScoreTracker.ts';
import { UnknownAgent } from './UnknownAgent.ts';

export function createPolicyDriverSystem(agents: Map<number, UnknownAgent>, { world } = GameDI) {
    const { Tank, Vehicle, VehicleController, Children } = getGameComponents(world);
    let frame = 0;

    return function updatePolicyDriver(_delta: number) {
        frame++;

        // Accumulate combat score every tick (hits/kills happen between decisions),
        // so calculateActionReward sees the right cumulative value at decision points.
        scoreTracker.update();

        const tanks = query(world, [Tank, Vehicle, VehicleController, Children]);

        let anyDecision = false;
        for (let i = 0; i < tanks.length; i++) {
            const eid = tanks[i];
            if (agents.has(eid) && needsDecision(eid)) { anyDecision = true; break; }
        }
        if (!anyDecision) return;

        // One board snapshot serves every observer this tick (uses GameDI.world).
        snapshotUnknownBoard();

        for (let i = 0; i < tanks.length; i++) {
            const eid = tanks[i];
            const agent = agents.get(eid);
            if (agent && needsDecision(eid)) agent.decide();
        }
    };
}
