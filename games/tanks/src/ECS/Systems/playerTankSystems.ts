import { DI } from '../../DI';
import { RigidBodyRef } from '../Components/Physical.ts';
import { RevoluteImpulseJoint, Vector2 } from '@dimforge/rapier2d';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { Tank, TankPart } from '../Components/Tank.ts';

export function createPlayerTankPositionSystem(tankId: number, { document, physicalWorld } = DI) {
    let speed = 0;
    let rotation = 0;

    const acceleration = 0.1; // Как быстро набирается скорость
    const maxSpeed = 1.5; // Максимальная скорость танка
    const rotationSpeed = 1; // Скорость поворота
    const impulseFactor = 1000000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 3000000; // Масштаб крутящего момента

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
    const inertialFactor = 0.002;


    const mousePosition = new Vector2(0, 0);
    document.addEventListener('mousemove', (event) => {
        mousePosition.x = event.clientX;
        mousePosition.y = event.clientY;
    });

    let currentTargetAngle = 0;

    function rotateByMotor(delta: number) {
        // Получаем RB танка и башни
        const turretEid = Tank.turretEId[tankEid];
        const jointPid = TankPart.jointPid[turretEid];
        const turretJoint = physicalWorld.getImpulseJoint(jointPid);
        const tankRB = physicalWorld.getRigidBody(RigidBodyRef.id[tankEid]);
        const turretRB = physicalWorld.getRigidBody(RigidBodyRef.id[Tank.turretEId[tankEid]]);
        const tankRot = tankRB.rotation();
        const turretPos = turretRB.translation();

        // Глобальный угол от точки поворота башни к позиции мыши
        const targetRot = Math.atan2(
            mousePosition.y - turretPos.y,
            mousePosition.x - turretPos.x,
        );

        // Желаемый угол башни относительно танка (при условии, что в состоянии покоя башня выровнена с танком)
        const targetDeltaRot = targetRot - tankRot + Math.PI / 2;
        const nextTargetAngle = lerpAngle(currentTargetAngle, targetDeltaRot, inertialFactor * delta) % (2 * Math.PI);
        (turretJoint as RevoluteImpulseJoint).configureMotorPosition((currentTargetAngle = nextTargetAngle), stiffness, damping);
    }

    function lerpAngle(a: number, b: number, t: number): number {
        let diff = b - a;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        return a + diff * t;
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
        const targetRot = Math.atan2(
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

