import { DI } from '../../DI';
import { RigidBodyRef } from '../Components/Physical.ts';
import { RevoluteImpulseJoint, Vector2 } from '@dimforge/rapier2d';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { Tank, TankPart } from '../Components/Tank.ts';
import { atan2 } from '../../../../../lib/math.ts';

export function createPlayerTankPositionSystem(tankId: number, { document, physicalWorld } = DI) {
    let speed = 0;
    let rotation = 0;

    const acceleration = 0.1; // Как быстро набирается скорость
    const maxSpeed = 1.5; // Максимальная скорость танка
    const rotationSpeed = 1; // Скорость поворота
    const impulseFactor = 10000000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 30000000; // Масштаб крутящего момента

    let moveDirection = 0;
    let rotationDirection = 0;

    const nextLinvel = new Vector2(0, 0);

    document.addEventListener('keydown', (event) => {
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                moveDirection = 1;
                event.preventDefault();
                break;
            }
            case 's':
            case 'ArrowDown': {
                moveDirection = -1;
                event.preventDefault();
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                rotationDirection = -1;
                event.preventDefault();
                break;
            }
            case 'd':
            case 'ArrowRight': {
                rotationDirection = 1;
                event.preventDefault();
                break;
            }
        }
    });

    document.addEventListener('keyup', (event) => {
        switch (event.key) {
            case 'w':
            case 's':
            case 'ArrowUp':
            case 'ArrowDown': {
                moveDirection = 0;
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                rotationDirection = 0;
                break;
            }
        }
    });

    return (delta: number) => {
        const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankId]);

        // Управление скоростью
        if (moveDirection !== 0) {
            speed += moveDirection * acceleration;
            speed = Math.max(-maxSpeed, Math.min(maxSpeed, speed));
        } else {
            speed *= 0.5; // Плавное замедление
        }

        // Управление поворотом
        if (rotationDirection !== 0) {
            rotation += rotationDirection * rotationSpeed;
        } else {
            rotation *= 0.5; // Плавное уменьшение поворота
        }

        // Получаем текущий угол поворота
        const nextRotation = rb.rotation();

        // Задаем направление движения
        nextLinvel.x = 0;
        nextLinvel.y = -speed * impulseFactor;
        applyRotationToVector(nextLinvel, nextLinvel, nextRotation);


        // Применяем импульс для движения
        nextLinvel.x *= delta;
        nextLinvel.y *= delta;
        rb.applyImpulse(nextLinvel, true);

        // Применяем крутящий момент для поворота
        rb.applyTorqueImpulse(delta * rotation * rotationImpulseFactor, true);
    };
}

export function createPlayerTankTurretRotationSystem(tankEid: number, { document, physicalWorld } = DI) {
    const damping = 0.2;   // коэффициент демпфирования
    const stiffness = 1e6; // коэффициент жесткости (подбирается опытным путем)

    const impulseFactor = 10; // Масштаб импульса (настраиваемый)
    const maxRotationSpeed = Math.PI * 0.8; // Максимальная скорость поворота


    const mousePosition = new Vector2(0, 0);
    document.addEventListener('mousemove', (event) => {
        mousePosition.x = event.clientX;
        mousePosition.y = event.clientY;
    });

    let currentTargetAngle = 0;

    function rotateByMotor(delta: number) {
        // Получаем данные для башни
        const turretEid = Tank.turretEId[tankEid];
        const jointPid = TankPart.jointPid[turretEid];
        const turretJoint = physicalWorld.getImpulseJoint(jointPid);
        if (!turretJoint) return;

        const tankRB = physicalWorld.getRigidBody(RigidBodyRef.id[tankEid]);
        const turretRB = physicalWorld.getRigidBody(RigidBodyRef.id[turretEid]);
        const tankRot = tankRB.rotation();
        const turretPos = turretRB.translation();

        // Глобальный угол от дула к позиции мыши (учитывая возможное смещение, например, из-за системы координат)
        const targetRot =
            Math.PI / 2 +
            Math.atan2(mousePosition.y - turretPos.y, mousePosition.x - turretPos.x);
        // Желаемый угол дула относительно танка
        const desiredDeltaRot = normalizeAngle(targetRot - tankRot);
        // Разница между текущим целевым углом и желаемым
        const rawDiff = normalizeAngle(desiredDeltaRot - currentTargetAngle);
        // Ограничение максимального изменения угла за кадр (delta приходит в мс, поэтому делим на 1000)
        const maxAngleChange = maxRotationSpeed * (delta / 1000);

        // Расстояние от мыши до дула
        const distance = Math.sqrt(
            (mousePosition.x - turretPos.x) ** 2 +
            (mousePosition.y - turretPos.y) ** 2,
        );

        // Плавно интерполируем влияние мыши от 0 до 1
        const influence = smoothStep(10, 100, distance);

        // Ограничиваем изменение угла с учётом влияния мыши
        const limitedDiff = Math.sign(rawDiff) * Math.min(Math.abs(rawDiff), maxAngleChange) * influence;

        // Обновляем текущий целевой угол (нормализуем для корректной работы)
        currentTargetAngle = normalizeAngle(currentTargetAngle + limitedDiff);

        // Применяем новый угол к мотору
        if ('configureMotorPosition' in turretJoint) {
            (turretJoint as RevoluteImpulseJoint).configureMotorPosition(
                currentTargetAngle,
                stiffness,
                damping,
            );
        }
    }

    // @ts-ignore
    function rotateByImpulse(delta: number) {
        // Получаем RB дула (башни) и родительского танка
        const tankRB = physicalWorld.getRigidBody(RigidBodyRef.id[tankEid]);
        const turretRB = physicalWorld.getRigidBody(RigidBodyRef.id[Tank.turretEId[tankEid]]);
        // Определяем углы: родителя и дула (в мировых координатах)
        const tankRot = tankRB.rotation();
        const turretRot = turretRB.rotation();
        const turretPos = turretRB.translation();
        // Вычисляем локальный угол дула относительно родителя
        const turretDeltaRot = turretRot - tankRot;
        // Глобальный угол от точки поворота к мыши
        const targetRot = atan2(
            mousePosition.y - turretPos.y,
            mousePosition.x - turretPos.x,
        );
        // Желаемый локальный угол дула относительно родителя
        const targetDeltaRot = targetRot - tankRot;
        // Вычисляем разницу углов (ошибку) и нормализуем в диапазоне [-π, π]
        const errorAngle = ((targetDeltaRot - turretDeltaRot + Math.PI * 1.5) % (2 * Math.PI)) - Math.PI;
        const torqueImpulse = stiffness * errorAngle - damping * turretRB.angvel();
        // Применяем импульс крутящего момента к RB дула
        turretRB.applyTorqueImpulse(delta * torqueImpulse * impulseFactor, true);
    }

    return (delta: number) => {
        rotateByMotor(delta);
        // rotateByImpulse(delta);
    };

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