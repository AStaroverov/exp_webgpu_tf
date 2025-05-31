import { TankAgent } from './CurrentActorAgent.ts';
import { OrnsteinUhlenbeckNoise } from '../../../../../lib/OrnsteinUhlenbeckNoise.ts';
import { Actions, applyActionToTank } from '../../TensorFlow/Common/applyActionToTank.ts';
import { RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { abs, atan2, hypot, lerp, normalizeAngle, PI } from '../../../../../lib/math.ts';
import { findTankEnemiesEids } from '../Utils/snapshotTankInputTensor.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';

export type SimpleBotFeatures = {
    move?: number;
    aim?: {
        aimError: number;
        shootChance: number;
    };
}

export class SimpleBot implements TankAgent {
    private waypoint?: { x: number; y: number };
    private ouNoiseRot = new OrnsteinUhlenbeckNoise(0, 0.15, 0.3);

    constructor(
        public readonly tankEid: number,
        private readonly features: SimpleBotFeatures = {},
    ) {
    }

    updateTankBehaviour(width: number, height: number): void {
        this.updateWaypoint(width, height);

        const targetId = this.withAim() ? this.getTarget() : undefined;
        const aim = targetId !== undefined ? this.getAimAction(targetId) : { shoot: -1, turretRot: 0 };
        const rotation = this.withMove() ? this.getRotationAction() : 0;
        const move = this.withMove() ? this.getMoveAction(rotation) : 0;

        const action: Actions = [
            aim.shoot,
            move * (this.features.move ?? 0),
            rotation * (this.features.move ?? 0),
            aim.turretRot,
        ];

        applyActionToTank(this.tankEid, action);
    }

    private withMove() {
        return this.features.move !== undefined;
    }

    private withAim() {
        return this.features.aim !== undefined;
    }

    private updateWaypoint(width: number, height: number): void {
        if (!this.waypoint) {
            this.waypoint = getRandomWaypoint(width, height);
        }

        const sx = RigidBodyState.position.get(this.tankEid, 0);
        const sy = RigidBodyState.position.get(this.tankEid, 1);
        const dx = this.waypoint.x - sx;
        const dy = this.waypoint.y - sy;

        if (hypot(dx, dy) < 100) {
            this.waypoint = getRandomWaypoint(width, height);
        }

    }

    private getTarget(): number | undefined {
        const enemies = findTankEnemiesEids(this.tankEid) as number[];
        const tx = RigidBodyState.position.get(this.tankEid, 0);
        const ty = RigidBodyState.position.get(this.tankEid, 1);
        let bestId: number | undefined;
        let bestDist = Infinity;

        for (const eid of enemies) {
            if (getTankHealth(eid) <= 0) continue;
            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const d = hypot(tx - ex, ty - ey);
            if (d < bestDist) {
                bestDist = d;
                bestId = eid;
            }
        }

        return bestId;
    }

    private getAimAction(targetId: number): { shoot: number, turretRot: number; } {
        const aimError = this.features.aim?.aimError ?? 0;
        const shootChance = this.features.aim?.shootChance ?? 0;

        const tankX = RigidBodyState.position.get(this.tankEid, 0);
        const tankY = RigidBodyState.position.get(this.tankEid, 1);
        const targetX = RigidBodyState.position.get(targetId, 0);
        const targetY = RigidBodyState.position.get(targetId, 1);

        const dx = targetX - tankX;
        const dy = targetY - tankY;

        const currentTurret = RigidBodyState.rotation[Tank.turretEId[this.tankEid]];   // [-π, π]
        const targetTurret = atan2(dy, dx);                        // [-π, π]

        let turretRot = normalizeAngle(targetTurret - currentTurret + PI / 2);
        const misalign = abs(turretRot) < 0.2;

        turretRot += this.ouNoiseRot.next() * aimError;
        turretRot = normalizeAngle(turretRot);

        const shoot = (misalign && shootChance > Math.random()) ? 1 : -1;

        return { turretRot, shoot };
    }

    private getMoveAction(rotation: number): number {
        return lerp(0.3, 1, 1 - abs(rotation));
    }

    private getRotationAction(): number {
        if (!this.waypoint) return 0;

        // --- позиция танка ---
        const px = RigidBodyState.position.get(this.tankEid, 0);
        const py = RigidBodyState.position.get(this.tankEid, 1);

        // --- позиция точки ---
        const dx = this.waypoint.x - px;
        const dy = this.waypoint.y - py;
        const targetAngle = atan2(dy, dx);

        // --- текущий угол корпуса ---
        const bodyAngle = RigidBodyState.rotation[this.tankEid];

        // --- разница углов, нормализованная к [-π, π] ---
        const delta = wrapPi(targetAngle - bodyAngle + PI / 2);    // -π … π

        // --- преобразуем в диапазон [-1; 1] ---
        return (delta / PI);                // -1 … 1
    }
}

function wrapPi(a: number): number {
    while (a > PI) a -= 2 * PI;
    while (a < -PI) a += 2 * PI;
    return a;
}

function getRandomWaypoint(width: number, height: number) {
    return {
        x: randomRangeFloat(200, width - 200),
        y: randomRangeFloat(200, height - 200),
    };
}
