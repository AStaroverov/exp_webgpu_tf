import { query } from "bitecs";
import { Vehicle } from "../../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../../Game/DI/GameDI";
import { RigidBodyState } from "../../Game/ECS/Components/Physical";
import { VehicleController } from "../../Game/ECS/Components/VehicleController";
import { PlayerRef } from "../../Game/ECS/Components/PlayerRef";
import { Score } from "../../Game/ECS/Components/Score";
import { Tank } from "../../Game/ECS/Components/Tank";
import { getTankHealth } from "../../Game/ECS/Entities/Tank/TankUtils";
import { findTankEnemiesEids } from "../Pilots/Utils/snapshotTankInputTensor";
import { computeObstacleGrid } from "../../../../ml-common/computeObstacleGrid";
import { computeConnectivityMap } from "../../../../ml-common/computeConnectivityMap";
import { GRID_SIZE } from "../../../../ml/src/Models/Create";

const NAVIGATION_COEFF = 0.003;
const AIM_COEFF = 0.005;
const MOVE_THRESHOLD = 0.1;
const SPEED_THRESHOLD = 1;

export function createMlScoreSystem({ world } = GameDI) {
    let connectivityMap: Float32Array | null = null;

    const tick = () => {
        if (!MLState.enabled) return;

        if (connectivityMap == null) {
            const obstacleGrid = computeObstacleGrid(world, GameDI.width, GameDI.height);
            connectivityMap = computeConnectivityMap(obstacleGrid);
        }

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];

            addNavigationReward(vehicleEid, playerId, connectivityMap);
            addAimReward(vehicleEid, playerId);
        }
    };

    const dispose = () => {
        connectivityMap = null;
    };

    return { tick, dispose };
}

function addNavigationReward(
    vehicleEid: number,
    playerId: number,
    connectivityMap: Float32Array,
): void {
    const move = VehicleController.move[vehicleEid];
    if (Math.abs(move) < MOVE_THRESHOLD) return;

    const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
    const vx = linvel[0];
    const vy = linvel[1];
    const speed = Math.sqrt(vx * vx + vy * vy);
    if (speed < SPEED_THRESHOLD) return;

    const cellW = GameDI.width / GRID_SIZE;
    const cellH = GameDI.height / GRID_SIZE;

    const px = RigidBodyState.position.get(vehicleEid, 0);
    const py = RigidBodyState.position.get(vehicleEid, 1);
    const col = Math.max(0, Math.min(GRID_SIZE - 1, (px / cellW) | 0));
    const row = Math.max(0, Math.min(GRID_SIZE - 1, (py / cellH) | 0));

    const targetCol = Math.max(0, Math.min(GRID_SIZE - 1, col + Math.sign(vx)));
    const targetRow = Math.max(0, Math.min(GRID_SIZE - 1, row + Math.sign(vy)));
    const targetIdx = targetRow * GRID_SIZE + targetCol;

    Score.addNavigation(playerId, connectivityMap[targetIdx] * Math.abs(move) * NAVIGATION_COEFF);
}

function addAimReward(
    vehicleEid: number,
    playerId: number,
): void {
    const turretEid = Tank.turretEId[vehicleEid];
    const turretRot = RigidBodyState.rotation[turretEid];

    // Turret forward direction
    const fwdX = Math.cos(turretRot);
    const fwdY = Math.sin(turretRot);

    const tx = RigidBodyState.position.get(vehicleEid, 0);
    const ty = RigidBodyState.position.get(vehicleEid, 1);

    // Find closest alive enemy
    const enemies = findTankEnemiesEids(vehicleEid);
    let bestCos = -1;
    let bestDistFactor = 0;

    for (let i = 0; i < enemies.length; i++) {
        const eid = enemies[i];
        if (getTankHealth(eid) <= 0) continue;

        const ex = RigidBodyState.position.get(eid, 0);
        const ey = RigidBodyState.position.get(eid, 1);
        const dx = ex - tx;
        const dy = ey - ty;
        const dist = Math.sqrt(dx * dx + dy * dy);
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
