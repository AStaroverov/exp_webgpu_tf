import { DI } from '../../DI';
import { TankController } from '../Components/TankController.ts';
import { RevoluteImpulseJoint, Vector2 } from '@dimforge/rapier2d';
import { defineQuery } from 'bitecs';
import { Tank, TankPart } from '../Components/Tank.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { sqrt } from '../../../../../lib/math.ts';

export function createTankPositionSystem({ world, physicalWorld } = DI) {
    const nextLinvel = new Vector2(0, 0);
    const impulseFactor = 15000000000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 100000000000; // Масштаб крутящего момента

    const tanksQuery = defineQuery([Tank, TankController]);

    return (delta: number) => {
        const tankPids = tanksQuery(world);
        for (let i = 0; i < tankPids.length; i++) {
            const tankId = tankPids[i];
            const moveDirection = TankController.move[tankId];
            const rotationDirection = TankController.rotation[tankId];
            const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankId]);
            const rotation = rb.rotation();

            // Задаем направление движения
            nextLinvel.x = 0;
            nextLinvel.y = -moveDirection * impulseFactor * delta / 1000;
            applyRotationToVector(nextLinvel, nextLinvel, rotation);

            // Применяем импульс для движения
            rb.applyImpulse(nextLinvel, true);
            // Применяем крутящий момент для поворота
            rb.applyTorqueImpulse(rotationDirection * rotationImpulseFactor * delta / 1000, true);
        }
    };
}


export function createTankTurretRotationSystem({ world } = DI) {
    const tankQuery = defineQuery([Tank, TankController]);

    return (delta: number) => {
        const tankPids = tankQuery(world);

        for (let i = 0; i < tankPids.length; i++) {
            const tankEid = tankPids[i];
            rotateByMotor(delta, tankEid);
        }
    };
}

const damping = 0.2;   // коэффициент демпфирования
const stiffness = 1e6; // коэффициент жесткости (подбирается опытным путем)
const maxRotationSpeed = Math.PI * 0.8; // Максимальная скорость поворота
function rotateByMotor(delta: number, tankEid: number, { physicalWorld } = DI) {
    // Получаем данные для башни
    const turretEid = Tank.turretEId[tankEid];
    const jointPid = TankPart.jointPid[turretEid];
    const turretJoint = physicalWorld.getImpulseJoint(jointPid) as RevoluteImpulseJoint;
    if (!turretJoint) return;

    const tankRB = physicalWorld.getRigidBody(RigidBodyRef.id[tankEid]);
    const turretRB = physicalWorld.getRigidBody(RigidBodyRef.id[turretEid]);

    const tankRot = tankRB.rotation();
    const turretRot = turretRB.rotation();
    const turretPos = turretRB.translation();
    const targetPos = TankController.turretTarget[tankEid];
    // Глобальный угол от дула к позиции цели
    const targetRot = Math.atan2(targetPos[1] - turretPos.y, targetPos[0] - turretPos.x) + Math.PI / 2;
    const relTurretRot = normalizeAngle(turretRot - tankRot);
    const relTargetTurretRot = normalizeAngle(targetRot - tankRot);
    const deltaRot = normalizeAngle(relTargetTurretRot - relTurretRot);
    // Расстояние от мыши до дула
    const distance = sqrt((targetPos[0] - turretPos.x) ** 2 + (targetPos[1] - turretPos.y) ** 2);
    // Плавно интерполируем влияние мыши от 0 до 1
    const influence = smoothStep(10, 100, distance);
    // Ограничиваем изменение угла с учётом влияния мыши
    const maxAngleChange = maxRotationSpeed * (delta / 1000);
    const limitedDeltaRot = Math.sign(deltaRot) * Math.min(Math.abs(deltaRot), maxAngleChange) * influence;
    // Применяем новый угол к мотору
    turretJoint.configureMotorPosition(
        normalizeAngle(relTurretRot + limitedDeltaRot),
        stiffness,
        damping,
    );
}

// Функция нормализации угла в диапазоне [-π, π]
function normalizeAngle(angle: number): number {
    while (angle < -Math.PI) angle += 2 * Math.PI;
    while (angle > Math.PI) angle -= 2 * Math.PI;
    return angle;
}

// Плавная функция smoothStep: значение 0, если x <= edge0, 1 если x >= edge1, и плавное переходное значение между ними.
function smoothStep(edge0: number, edge1: number, x: number): number {
    // Нормализуем x к диапазону [0,1]
    const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}