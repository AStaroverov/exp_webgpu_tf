import { query } from "bitecs";
import { RingBuffer } from "ring-buffer-ts";
import { Vehicle } from "../../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../../Game/DI/GameDI";
import { RigidBodyState } from "../../Game/ECS/Components/Physical";
import { PlayerRef } from "../../Game/ECS/Components/PlayerRef";
import { Score } from "../../Game/ECS/Components/Score";
import { Tank } from "../../Game/ECS/Components/Tank";
import { getTankHealth } from "../../Game/ECS/Entities/Tank/TankUtils";
import { findTankEnemiesEids, findVehicleFromPart } from "../Pilots/Utils/snapshotTankInputTensor";
import { TankInputTensor, RAY_BUFFER, RAYS_COUNT, RayHitType } from "../Pilots/Components/TankState";
import { SNAPSHOT_EVERY } from "../../../../ml-common/consts";
import { cos, sin, sqrt, hypot } from "../../../../../lib/math";
import { clamp } from "lodash";

// All coefficients are per-action (1 action = SNAPSHOT_EVERY ticks ≈ 200ms)
const ENGAGED_RAY_THRESHOLD = 2;

const AIM_COEFF = 0.001;

const FOCUS_REWARD = 0.0001;
const FOCUS_PENALTY = -0.0005;

const MOVEMENT_COEFF = 0.1;
const MOVEMENT_ACTIONS = 10; // ~2sec window
const MOVEMENT_DIST_THRESHOLD = 500;

export function createMlScoreSystem({ world } = GameDI) {
    let frame = 0;

    const tick = () => {
        if (!MLState.enabled) return;

        // Evaluate only once per action (aligned with agent decision step)
        if (frame++ % SNAPSHOT_EVERY !== 0) return;

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];
            const enemies = findTankEnemiesEids(vehicleEid);
            const enemyRayHits = collectEnemyRayHits(vehicleEid);
            addAimReward(vehicleEid, playerId, enemies, enemyRayHits);
            addMovementReward(vehicleEid, playerId);
            addEnemyFocusReward(playerId, enemyRayHits);
        }
    };

    const idleRings = new Map<number, RingBuffer<{ x: number, y: number }>>();

    const dispose = () => {
        frame = 0;
        idleRings.clear();
    };

    function addEnemyFocusReward(
        playerId: number,
        enemyRayHits: Map<number, number>,
    ): void {
        let maxRays = 0;
        for (const count of enemyRayHits.values()) {
            if (count > maxRays) maxRays = count;
            if (maxRays >= 3) break;
        }

        if (maxRays >= 3) {
            Score.addEnemyFocus(playerId, FOCUS_REWARD);
        } else if (maxRays < 2) {
            Score.addEnemyUnfocus(playerId, FOCUS_PENALTY);
        }
    }

    function addMovementReward(vehicleEid: number, playerId: number): void {
        const px = RigidBodyState.position.get(vehicleEid, 0);
        const py = RigidBodyState.position.get(vehicleEid, 1);

        let ring = idleRings.get(vehicleEid);
        if (!ring) {
            ring = new RingBuffer<{ x: number, y: number }>(MOVEMENT_ACTIONS);
            idleRings.set(vehicleEid, ring);
        }

        ring.add({ x: px, y: py });

        if (!ring.isFull()) return;
        const oldest = ring.getFirst();
        if (!oldest) return;
        const displacement = hypot(px - oldest.x, py - oldest.y);
        const reward = clamp(displacement / MOVEMENT_DIST_THRESHOLD, 0, 1);
        Score.addMovement(playerId, MOVEMENT_COEFF * (reward ** 2));
    }

    function addAimReward(
        vehicleEid: number,
        playerId: number,
        enemies: number[],
        enemyRayHits: Map<number, number>,
    ): void {
        const turretEid = Tank.turretEId[vehicleEid];
        const turretRot = RigidBodyState.rotation[turretEid];

        const fwdX = cos(turretRot);
        const fwdY = sin(turretRot);

        const tx = RigidBodyState.position.get(vehicleEid, 0);
        const ty = RigidBodyState.position.get(vehicleEid, 1);

        let bestCos = -1;
        let bestDistFactor = 0;

        for (let i = 0; i < enemies.length; i++) {
            const eid = enemies[i];
            if (getTankHealth(eid) <= 0) continue;
            // Only reward aiming at enemies clearly visible (2+ rays), not through cracks
            if ((enemyRayHits.get(eid) ?? 0) < ENGAGED_RAY_THRESHOLD) continue;

            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const dx = ex - tx;
            const dy = ey - ty;
            const dist = sqrt(dx * dx + dy * dy);
            if (dist < 1) continue;

            const cosAngle = (fwdX * dx + fwdY * dy) / dist;
            const distFactor = 1 / (1 + dist / (GameDI.width * 0.5));

            if (cosAngle * distFactor > bestCos * bestDistFactor) {
                bestCos = cosAngle;
                bestDistFactor = distFactor;
            }
        }

        if (bestCos <= 0) return;

        Score.addAimAlignment(playerId, bestCos * bestDistFactor * AIM_COEFF);
    }

    return { tick, dispose };
}


function collectEnemyRayHits(vehicleEid: number): Map<number, number> {
    const hits = new Map<number, number>();
    for (let i = 0; i < RAYS_COUNT; i++) {
        const offset = i * RAY_BUFFER;
        if (TankInputTensor.raysData.get(vehicleEid, offset) !== RayHitType.ENEMY_VEHICLE) continue;
        const partEid = TankInputTensor.raysData.get(vehicleEid, offset + 1);
        const ownerEid = findVehicleFromPart(partEid);
        if (ownerEid === 0) continue;
        hits.set(ownerEid, (hits.get(ownerEid) ?? 0) + 1);
    }
    return hits;
}
