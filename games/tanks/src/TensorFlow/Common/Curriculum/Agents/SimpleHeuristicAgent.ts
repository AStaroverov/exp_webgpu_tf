import { RigidBodyState } from '../../../../ECS/Components/Physical';
import { hypot, max, min } from '../../../../../../../lib/math.ts';
import { TankController } from '../../../../ECS/Components/TankController.ts';
import { getTankHealth } from '../../../../ECS/Entities/Tank/TankUtils.ts';
import { findTankEnemiesEids } from '../../../../ECS/Systems/RL/createTankInputTensorSystem.ts';
import { Actions } from '../../actions.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { TankAgent } from './ActorAgent.ts';
import { randomRangeFloat, randomSign } from '../../../../../../../lib/random.ts';

interface BodyState {
    rotation: -1 | 0 | 1;
    escapingBorder: boolean;
    borderRotation: -1 | 1;
}

export type AgentFeatures = {
    move?: number;
    aim?: boolean;
    shoot?: boolean;
}

export class SimpleHeuristicAgent implements TankAgent {
    private tickCounter = 0;
    private bodyState: BodyState = { rotation: 0, escapingBorder: false, borderRotation: 1 };
    private currentTargetId: number | undefined;

    constructor(
        public readonly tankEid: number,
        private readonly features: AgentFeatures = {},
        private readonly turnInterval: number = 30,
        private readonly shootChance: number = 0.25,
        private readonly border: number = 200,
    ) {
    }

    updateTankBehaviour(width: number, height: number): void {
        this.tickCounter++;

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
        const rotation = withMove ? this.updateBodyRotation(width, height) : 0;

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

    private updateBodyRotation(width: number, height: number): -1 | 0 | 1 {
        const x = RigidBodyState.position.get(this.tankEid, 0);
        const y = RigidBodyState.position.get(this.tankEid, 1);
        const atBorder = (
            x < this.border || x > width - this.border || y < this.border || y > height - this.border
        );

        if (atBorder) {
            if (!this.bodyState.escapingBorder) {
                this.bodyState.escapingBorder = true;
                this.bodyState.borderRotation = Math.random() < 0.5 ? -1 : 1;
            }
            this.bodyState.rotation = this.bodyState.borderRotation;
            return this.bodyState.rotation;
        }

        // вышли из зоны края
        if (this.bodyState.escapingBorder) {
            this.bodyState.escapingBorder = false;
        }

        // случайная смена курса, если включена фича randomTurn
        if (this.tickCounter % this.turnInterval === 0) {
            const r = Math.random();
            this.bodyState.rotation = r < 0.33 ? -1 : r < 0.66 ? 1 : 0;
        }

        return this.bodyState.rotation;
    }
}
