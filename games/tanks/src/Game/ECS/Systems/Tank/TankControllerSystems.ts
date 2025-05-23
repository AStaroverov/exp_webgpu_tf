import { GameDI } from '../../../DI/GameDI.ts';
import { TankController } from '../../Components/TankController.ts';
import { RevoluteImpulseJoint, Vector2 } from '@dimforge/rapier2d-simd';
import { Tank } from '../../Components/Tank.ts';
import { RigidBodyRef, RigidBodyState } from '../../Components/Physical.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { query } from 'bitecs';
import { TankPart } from '../../Components/TankPart.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';

export function createTankPositionSystem({ world, physicalWorld } = GameDI) {
    const nextLinvel = new Vector2(0, 0);
    const impulseFactor = 15000000000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 100000000000; // Масштаб крутящего момента

    return (delta: number) => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankId = tankEids[i];
            const moveDirection = TankController.move[tankId];
            const rotationDirection = TankController.rotation[tankId];

            if (moveDirection === 0 && rotationDirection === 0) continue;

            const rotation = RigidBodyState.rotation[tankId];
            // Задаем направление движения
            nextLinvel.x = 0;
            nextLinvel.y = -moveDirection * impulseFactor * delta / 1000;
            applyRotationToVector(nextLinvel, nextLinvel, rotation);

            const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankId]);
            // Применяем импульс для движения
            rb.applyImpulse(nextLinvel, true);
            // Применяем крутящий момент для поворота
            rb.applyTorqueImpulse(rotationDirection * rotationImpulseFactor * delta / 1000, true);
        }
    };
}

export function createTankTurretRotationSystem({ world } = GameDI) {
    return (delta: number) => {
        const tankPids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankPids.length; i++) {
            const tankEid = tankPids[i];
            rotateByMotor(delta, tankEid);
        }
    };
}

const damping = 0.2;   // коэффициент демпфирования
const stiffness = 1e6; // коэффициент жесткости (подбирается опытным путем)
const maxRotationSpeed = Math.PI * 0.8; // Максимальная скорость поворота
function rotateByMotor(delta: number, tankEid: number, { physicalWorld } = GameDI) {
    // Получаем данные для башни
    const turretEid = Tank.turretEId[tankEid];
    const jointPid = TankPart.jointPid[turretEid];
    const turretJoint = physicalWorld.getImpulseJoint(jointPid) as RevoluteImpulseJoint;
    if (!turretJoint) return;

    const tankRot = RigidBodyState.rotation[tankEid];
    const turretRot = RigidBodyState.rotation[turretEid];
    const turretRotDir = TankController.turretRotation[tankEid];

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
