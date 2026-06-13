/**
 * createPolicyDriverSystem — the ML decision driver, installed as a
 * `SystemGroup.Before` plugin IN PLACE OF the game's `createStandInDriverSystem`.
 *
 * Per tick: find living agent-controlled tanks that present an open slot
 * (`needsDecision`), refresh the board observation once for all of them, then KICK
 * OFF each agent's decision. Inference is ASYNC (WebGPU readback), so `decide()`
 * returns a promise that enqueues the action when it resolves — the driver does not
 * block the tick. An in-flight set gates re-entry so a tank that is mid-decision is
 * not asked again before its action lands.
 *
 * Two consumers, one driver:
 *   - actor (headless, deterministic): the episode manager `await`s `drain()` after
 *     each tick, so the decision kicked off in tick N is enqueued before tick N+1's
 *     `updateActions` consumes the queue — exactly the 1-tick pipeline the sync
 *     driver already had.
 *   - vis tab (rendered, FPS-sensitive): never drains — fire-and-forget. The action
 *     just lands a tick or two later; the `MAX_QUEUE` slack absorbs the latency.
 */

import { query } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { needsDecision } from "../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts";
import { snapshotUnknownBoard } from "../state/snapshotUnknownBoard.ts";
import { scoreTracker } from "../reward/ScoreTracker.ts";

/**
 * Anything the driver can poke at a decision point: a learning `UnknownAgent`, a
 * `FrozenAgent` (both async), or a scripted `RandomBot` (sync). Keeping it structural
 * lets one driver loop serve all three — `decide()` may return void or a promise.
 */
export type TankDriver = { decide(): void | Promise<void> };

export function createPolicyDriverSystem(agents: Map<number, TankDriver>, { world } = GameDI) {
  const { Tank, Vehicle, VehicleController, Children } = getGameComponents(world);

  // Tanks with a decision in flight (kicked off, action not yet enqueued).
  const inFlight = new Set<number>();
  // Self-cleaning so the fire-and-forget vis path never leaks: each promise removes
  // itself on settle, and `drain()` awaits whatever is currently outstanding.
  const pending = new Set<Promise<void>>();

  const system = function updatePolicyDriver(_delta: number) {
    // Accumulate combat score every tick (hits/kills happen between decisions),
    // so calculateActionReward sees the right cumulative value at decision points.
    scoreTracker.update();

    const tanks = query(world, [Tank, Vehicle, VehicleController, Children]);

    let anyDecision = false;
    for (let i = 0; i < tanks.length; i++) {
      const eid = tanks[i];
      if (agents.has(eid) && !inFlight.has(eid) && needsDecision(eid)) {
        anyDecision = true;
        break;
      }
    }
    if (!anyDecision) return;

    // One board snapshot serves every observer this tick (uses GameDI.world). Each
    // agent reads its row synchronously inside decide() before awaiting, so the
    // snapshot is never read concurrently with the next tick's write.
    snapshotUnknownBoard();

    for (let i = 0; i < tanks.length; i++) {
      const eid = tanks[i];
      const agent = agents.get(eid);
      if (!agent || inFlight.has(eid) || !needsDecision(eid)) continue;

      inFlight.add(eid);
      let promise: Promise<void>;
      promise = Promise.resolve(agent.decide())
        .catch((err) => {
          console.error("Policy decide failed:", err);
        })
        .finally(() => {
          inFlight.delete(eid);
          pending.delete(promise);
        });
      pending.add(promise);
    }
  };

  /** Await every decision currently in flight (actor path; vis never calls this). */
  const drain = async function drainDecisions(): Promise<void> {
    if (pending.size === 0) return;
    await Promise.all(pending);
  };

  return { system, drain };
}
