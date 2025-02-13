import { DI } from '../../DI';
import { RigidBodyRef } from '../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';

export function createUpdatePlayerTankPositionSystem(tankId: number, { document, physicalWorld } = DI) {
    let speed = 0;
    let rotation = 0;

    const acceleration = 0.1; // Как быстро набирается скорость
    const maxSpeed = 1.5; // Максимальная скорость танка
    const rotationSpeed = 0.003; // Скорость поворота
    const impulseFactor = 500000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 1000000000; // Масштаб крутящего момента

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

    return function () {
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
        rb.applyImpulse(nextLinvel, true);

        // Применяем крутящий момент для поворота
        rb.applyTorqueImpulse(rotation * rotationImpulseFactor, true);
    };
}
