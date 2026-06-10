import { query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { cos, sin } from "../../../../../../lib/math.ts";

const linvelVec = { x: 0, y: 0 };

/**
 * Rotates the velocity of every `Wander` body by a per-entity sinusoidal turn
 * rate, bending straight trajectories into smooth meandering curves (flame
 * tongues, drifting cold wisps). Speed magnitude is preserved — only the
 * direction wanders. Runs before the physics step so the curved velocity is
 * what the step integrates.
 */
export function createWanderSystem({ world, physicalWorld } = GameDI) {
  const { Wander, RigidBodyRef } = getGameComponents(world);

  return (delta: number) => {
    const eids = query(world, [Wander, RigidBodyRef]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];
      Wander.ageMs[eid] += delta;

      const pid = RigidBodyRef.id[eid];
      if (pid === 0) continue;
      const rb = physicalWorld.getRigidBody(pid);
      if (rb == null) continue;

      // Two detuned harmonics — a single sine reads as mechanical sway.
      const arg = Wander.ageMs[eid] * Wander.frequency[eid] + Wander.phase[eid];
      const omega = (sin(arg) + 0.5 * sin(arg * 2.3 + 1.7)) * Wander.angularSpeed[eid];
      const turn = omega * (delta / 1000);

      const v = rb.linvel();
      const ca = cos(turn);
      const sa = sin(turn);
      linvelVec.x = v.x * ca - v.y * sa;
      linvelVec.y = v.x * sa + v.y * ca;
      rb.setLinvel(linvelVec, true);
    }
  };
}
