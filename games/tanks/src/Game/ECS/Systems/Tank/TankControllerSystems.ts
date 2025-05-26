import { GameDI } from '../../../DI/GameDI.ts';
import { TankController } from '../../Components/TankController.ts';
import { RevoluteImpulseJoint, Vector2 } from '@dimforge/rapier2d-simd';
import { Tank } from '../../Components/Tank.ts';
import { RigidBodyRef, RigidBodyState } from '../../Components/Physical.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { query } from 'bitecs';
import { TankPart } from '../../Components/TankPart.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { TankTurret } from '../../Components/TankTurret.ts';

export enum TankEngineType {
    v6,
    v8,
    v12
}

export const mapTankEngineLabel = {
    [TankEngineType.v6]: 'v6',
    [TankEngineType.v8]: 'v8',
    [TankEngineType.v12]: 'v12',
};

const IMPULSE_FACTOR = 15000000000;
const ROTATION_IMPULSE_FACTOR = 150000000000;
const mapTypeToFeatures = {
    [TankEngineType.v6]: {
        impulseFactor: IMPULSE_FACTOR * 0.8,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR * 0.9,
    },
    [TankEngineType.v8]: {
        impulseFactor: IMPULSE_FACTOR,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR,
    },
    [TankEngineType.v12]: {
        impulseFactor: IMPULSE_FACTOR * 2,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR * 3,
    },
};

export function createTankPositionSystem({ world, physicalWorld } = GameDI) {
    const nextLinvel = new Vector2(0, 0);

    return (delta: number) => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const moveDirection = TankController.move[tankEid];
            const rotationDirection = TankController.rotation[tankEid];
            const {
                impulseFactor,
                rotationImpulseFactor,
            } = mapTypeToFeatures[Tank.engineType[tankEid] as TankEngineType];

            if (moveDirection === 0 && rotationDirection === 0) continue;

            const rotation = RigidBodyState.rotation[tankEid];
            // Задаем направление движения
            nextLinvel.x = 0;
            nextLinvel.y = -moveDirection * impulseFactor * delta / 1000;
            applyRotationToVector(nextLinvel, nextLinvel, rotation);

            const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankEid]);
            // Применяем импульс для движения
            rb.applyImpulse(nextLinvel, false);
            // Применяем крутящий момент для поворота
            rb.applyTorqueImpulse(rotationDirection * rotationImpulseFactor * delta / 1000, false);
        }
    };
}

export function createTankTurretRotationSystem({ world } = GameDI) {
    return (delta: number) => {
        const tankPids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankPids.length; i++) {
            rotateByMotor(tankPids[i], delta);
        }
    };
}

const damping = 0.2;   // коэффициент демпфирования
const stiffness = 1e6; // коэффициент жесткости (подбирается опытным путем)
function rotateByMotor(tankEid: number, delta: number, { physicalWorld } = GameDI) {
    // Получаем данные для башни
    const turretEid = Tank.turretEId[tankEid];
    const jointPid = TankPart.jointPid[turretEid];
    const turretJoint = physicalWorld.getImpulseJoint(jointPid) as RevoluteImpulseJoint;
    if (!turretJoint) return;

    const tankRot = RigidBodyState.rotation[tankEid];
    const turretRot = RigidBodyState.rotation[turretEid];
    const turretRotDir = TankController.turretRotation[tankEid];
    const maxRotationSpeed = TankTurret.rotationSpeed[turretEid];

    // Глобальный угол от дула к позиции цели
    const relTurretRot = normalizeAngle(turretRot - tankRot);
    // Ограничиваем изменение угла с учётом влияния мыши
    const deltaRot = turretRotDir * maxRotationSpeed * (delta / 1000);
    // Применяем новый угол к мотору
    turretJoint.configureMotorPosition(
        normalizeAngle(relTurretRot + deltaRot),
        stiffness,
        damping,
    );
}
