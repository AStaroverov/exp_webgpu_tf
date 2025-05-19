import { RigidBodyState } from '../../../../Game/ECS/Components/Physical';
import { abs, hypot, lerp, min } from '../../../../../../../lib/math.ts';
import { getAimPosition, getTankHealth } from '../../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { findTankEnemiesEids } from '../../../../Game/ECS/Utils/snapshotTankInputTensor.ts';
import { Actions, applyActionToTank } from '../../applyActionToTank.ts';
import { TankAgent } from './CurrentActorAgent.ts';
import { random, randomRangeFloat } from '../../../../../../../lib/random.ts';
import { OrnsteinUhlenbeckNoise } from '../../../../../../../lib/OrnsteinUhlenbeckNoise.ts';

export type SimpleHeuristicAgentFeatures = {
    move?: number;
    aim?: {
        aimError: number;
        shootChance: number;
    };
}

export class SimpleHeuristicAgent implements TankAgent {
    private waypoint?: { x: number; y: number };
    private ouNoiseX = new OrnsteinUhlenbeckNoise(0, 0.15, 0.3);
    private ouNoiseY = new OrnsteinUhlenbeckNoise(0, 0.15, 0.3);

    constructor(
        public readonly tankEid: number,
        private readonly features: SimpleHeuristicAgentFeatures = {},
    ) {
    }

    updateTankBehaviour(width: number, height: number): void {
        this.updateWaypoint(width, height);

        const targetId = this.withAim() ? this.getTarget() : undefined;
        const aim = targetId !== undefined ? this.getAimAction(targetId) : { aimX: 0, aimY: 0, shoot: -1 };
        const rotation = this.withMove() ? this.getRotationAction() : 0;
        const move = this.withMove() ? this.getMoveAction(rotation) : 0;

        const action: Actions = [
            aim.shoot,
            move,
            rotation,
            aim.aimX,
            aim.aimY,
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

    private getAimAction(targetId: number): { aimX: number; aimY: number; shoot: number } {
        const aimError = this.features.aim?.aimError ?? 0;
        const shootChance = this.features.aim?.shootChance ?? 0;

        const [x0, y0] = getAimPosition(this.tankEid);
        const x1 = RigidBodyState.position.get(targetId, 0);
        const y1 = RigidBodyState.position.get(targetId, 1);

        const dx = x1 - x0;
        const dy = y1 - y0;
        const misalign = abs(dx) + abs(dy);
        const shoot = ((misalign < 100 && shootChance > random()) || random() > 0.9) ? 1 : -1;

        const dist = hypot(dx, dy);
        const step = min(dist, 1);
        const baseX = dx / dist * step;
        const baseY = dy / dist * step;

        const noiseX = this.ouNoiseX.next() * aimError;
        const noiseY = this.ouNoiseY.next() * aimError;

        return {
            shoot,
            aimX: baseX + noiseX,
            aimY: baseY + noiseY,
        };
    }


    private getMoveAction(rotation: number): number {
        let move = randomRangeFloat(this.features.move ?? 0, 1);

        return move * lerp(0.3, 1, 1 - abs(rotation));
    }

    private getRotationAction(): number {
        if (!this.waypoint) return 0;

        // --- позиция танка ---
        const px = RigidBodyState.position.get(this.tankEid, 0);
        const py = RigidBodyState.position.get(this.tankEid, 1);

        // --- позиция точки ---
        const dx = this.waypoint.x - px;
        const dy = this.waypoint.y - py;
        const targetAngle = Math.atan2(dy, dx);

        // --- текущий угол корпуса ---
        const bodyAngle = RigidBodyState.rotation[this.tankEid];

        // --- разница углов, нормализованная к [-π, π] ---
        const delta = wrapPi(targetAngle - bodyAngle + Math.PI / 2);    // -π … π

        // --- преобразуем в диапазон [-1; 1] ---
        return (delta / Math.PI);                // -1 … 1
    }
}

function wrapPi(a: number): number {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
}

function getRandomWaypoint(width: number, height: number) {
    return {
        x: randomRangeFloat(200, width - 200),
        y: randomRangeFloat(200, height - 200),
    };
}
