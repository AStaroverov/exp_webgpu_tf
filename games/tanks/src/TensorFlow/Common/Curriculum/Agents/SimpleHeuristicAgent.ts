import { RigidBodyState } from '../../../../ECS/Components/Physical';
import { hypot, max, min } from '../../../../../../../lib/math.ts';
import { TankController } from '../../../../ECS/Components/TankController.ts';
import { getTankHealth } from '../../../../ECS/Entities/Tank/TankUtils.ts';
import { findTankEnemiesEids } from '../../../../ECS/Systems/RL/createTankInputTensorSystem.ts';
import { Actions } from '../../actions.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { TankAgent } from './ActorAgent.ts';
import { randomRangeFloat, randomSign } from '../../../../../../../lib/random.ts';

export type AgentFeatures = {
    move?: number;
    aim?: boolean;
    shoot?: boolean;
}

export class SimpleHeuristicAgent implements TankAgent {
    private waypoint?: { x: number; y: number };
    private currentTargetId: number | undefined;

    constructor(
        public readonly tankEid: number,
        private readonly features: AgentFeatures = {},
        private readonly shootChance: number = 0.25,
    ) {
    }

    updateTankBehaviour(width: number, height: number): void {
        this.updateWaypoint(width, height);

        // 1. выбор цели (если нужно прицеливаться или стрелять)
        if (this.features.aim === true || this.features.shoot === true) {
            this.currentTargetId = this.selectTarget();
        } else {
            this.currentTargetId = undefined;
        }

        // 2. управление башней
        const turretCmd = this.features.aim === false && this.features.shoot === false
            ? { aimX: 0 as -1 | 0 | 1, aimY: 0 as -1 | 0 | 1, shoot: 0 as 0 | 1 }
            : this.controlTurret(this.currentTargetId);

        // 3. управление корпусом
        const withMove = typeof this.features.move === 'number';
        const maxMove = withMove ? this.features.move : 0;
        const move = this.updateBodyVelocity();
        const rotation = withMove ? this.updateBodyRotationTowardsWaypoint() : 0;

        // 4. формируем действие
        const shoot = this.features.shoot === false ? 0 : turretCmd.shoot;

        const action: Actions = [
            shoot,
            move,
            rotation,
            turretCmd.aimX,
            turretCmd.aimY,
        ];
        applyActionToTank(this.tankEid, action, maxMove);
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

    private selectTarget(): number | undefined {
        const enemies = findTankEnemiesEids(this.tankEid) as number[];
        let bestId: number | undefined;
        let bestDist = Infinity;
        const sx = RigidBodyState.position.get(this.tankEid, 0);
        const sy = RigidBodyState.position.get(this.tankEid, 1);

        for (const eid of enemies) {
            if (getTankHealth(eid) <= 0) continue;
            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const d = hypot(sx - ex, sy - ey);
            if (d < bestDist) {
                bestDist = d;
                bestId = eid;
            }
        }
        return bestId;
    }

    private controlTurret(targetId: number | undefined): { aimX: -1 | 0 | 1; aimY: -1 | 0 | 1; shoot: 0 | 1 } {
        let aimX: -1 | 0 | 1 = 0;
        let aimY: -1 | 0 | 1 = 0;
        let shoot: 0 | 1 = 0;

        if (targetId !== undefined) {
            const sx = RigidBodyState.position.get(this.tankEid, 0);
            const sy = RigidBodyState.position.get(this.tankEid, 1);
            const ex = RigidBodyState.position.get(targetId, 0);
            const ey = RigidBodyState.position.get(targetId, 1);
            const tx = ex - sx;
            const ty = ey - sy;
            const len = hypot(tx, ty) + 1e-6;
            const tnx = tx / len;
            const tny = ty / len;

            const cx = TankController.turretDir.get(this.tankEid, 0);
            const cy = TankController.turretDir.get(this.tankEid, 1);

            const dot = max(-1, min(1, cx * tnx + cy * tny));
            const aligned = dot > 0.985;
            const cross = cx * tny - cy * tnx;

            if (this.features.aim === true && !aligned) {
                aimX = cross > 0 ? 1 : -1;
            }

            if (aligned && this.features.shoot === true && Math.random() < this.shootChance) {
                shoot = 1;
            }
        }

        return { aimX, aimY, shoot };
    }

    private updateBodyVelocity(): number {
        let move = TankController.move[this.tankEid];
        let velocity = randomRangeFloat(0, 1);

        if (velocity < move) {
            velocity *= randomSign();
        }

        return velocity;
    }

    private updateBodyRotationTowardsWaypoint(): number {
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
        const delta = -wrapPi((targetAngle - bodyAngle) - Math.PI / 2);    // -π … π

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
