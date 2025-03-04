import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { random, randomSign } from '../../../../../lib/random.ts';
import { TANK_RADIUS } from '../Common/consts.ts';
import { smoothstep } from '../../../../../lib/math.ts';

/**
 * Генерация управляемых случайных действий с элементами простых стратегий
 * @param tankId ID танка
 * @returns Массив с действием [shoot, move, rotate, aimX, aimY]
 */
export function generateGuidedRandomAction(tankId: number): number[] {
    // Получаем данные о танке
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    // Базовые случайные значения для движения и вращения
    let moveValue = (random() * 1.6 - 0.8); // от -0.8 до 0.8
    let rotateValue = (random() * 1.6 - 0.8); // от -0.8 до 0.8

    // Значения для прицеливания
    let aimX = 0;
    let aimY = 0;

    // Случайно выбираем стратегию поведения
    const behaviorStrategy = random();

    if (behaviorStrategy < 0.4) {
        // Стратегия: поиск и отслеживание ближайшего врага
        const nearestEnemy = findNearestEnemy(tankId);

        if (nearestEnemy) {
            // Устанавливаем прицел на врага с небольшим случайным отклонением
            aimX = nearestEnemy.x + (random() * 0.3 - 0.15);
            aimY = nearestEnemy.y + (random() * 0.3 - 0.15);

            // Вычисляем направление к врагу для движения
            const distToEnemy = Math.hypot(nearestEnemy.x - tankX, nearestEnemy.y - tankY);

            // Если враг слишком близко, отъезжаем, иначе подъезжаем на среднюю дистанцию
            if (distToEnemy < TANK_RADIUS * 3) {
                // Отъезжаем от врага
                moveValue = -0.5 - random() * 0.5; // от -0.5 до -1.0
            } else if (distToEnemy > TANK_RADIUS * 10) {
                // Подъезжаем к врагу
                moveValue = 0.5 + random() * 0.5; // от 0.5 до 1.0
            } else {
                // Поддерживаем дистанцию, случайное движение
                moveValue = random() * 1.6 - 0.8; // от -0.8 до 0.8
            }

            // Стреляем с большей вероятностью при хорошем прицеливании
            const shootValue = random() < 0.7 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        }
    } else if (behaviorStrategy < 0.7) {
        // Стратегия: патрулирование с периодическим сканированием
        // Движемся и поворачиваемся более последовательно
        moveValue = 0.5 + random() * 0.5; // от 0.5 до 1.0
        rotateValue = (random() - 0.5) * 0.6; // от -0.3 до 0.3 (меньше поворотов)

        // Прицеливаемся в разные части карты, но более плавно
        aimX = random() * 2 - 1;
        aimY = random() * 2 - 1;

        // Меньше стреляем при патрулировании
        const shootValue = random() < 0.3 ? 1 : 0;

        return [shootValue, moveValue, rotateValue, aimX, aimY];
    } else {
        // Стратегия: случайное движение, но сканирование потенциальных угроз
        // Ищем ближайшую пулю и уворачиваемся
        const nearestBullet = findNearestBullet(tankId);

        if (nearestBullet && nearestBullet.danger > 0.3) {
            // Есть опасная пуля - уворачиваемся

            // Выбираем направление уворота случайно
            const dodgeDir = random() > 0.5 ? 1 : -1;

            // Ставим более высокую скорость для уворота
            moveValue = 0.8 + random() * 0.2; // от 0.8 до 1.0

            // Поворачиваем в направлении движения
            rotateValue = dodgeDir * (0.5 + random() * 0.5); // от 0.5 до 1.0 с нужным знаком

            // Прицеливаемся в направлении, откуда пришла пуля
            aimX = nearestBullet.x + nearestBullet.vx * -5; // -5 это коэффициент "заглядывания назад"
            aimY = nearestBullet.y + nearestBullet.vy * -5;

            // Нормализуем координаты прицела
            const aimLen = Math.hypot(aimX, aimY);
            if (aimLen > 0) {
                aimX = aimX / aimLen;
                aimY = aimY / aimLen;
            }

            // Редко стреляем при уворачивании
            const shootValue = random() < 0.2 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        } else {
            // Нет опасности - случайные действия с небольшим смещением в сторону разумности
            moveValue = random() * 1.6 - 0.3; // от -0.3 до 1.3 (смещение вперёд)
            rotateValue = (random() - 0.5) * 1.4; // от -0.7 до 0.7

            // Прицеливаемся с большей вероятностью в углы и края карты,
            // где часто прячутся другие танки
            if (random() < 0.4) {
                // Выбираем один из углов или краёв
                const edgeX = random() < 0.5 ? -0.9 : 0.9;
                const edgeY = random() < 0.5 ? -0.9 : 0.9;

                // С небольшим случайным отклонением
                aimX = edgeX + (random() * 0.2 - 0.1);
                aimY = edgeY + (random() * 0.2 - 0.1);
            } else {
                // Просто случайное направление прицела
                aimX = random() * 2 - 1;
                aimY = random() * 2 - 1;
            }

            // Стреляем в случайные моменты
            const shootValue = random() < 0.4 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        }
    }

    // Запасной вариант - полностью случайные действия
    return generatePureRandomAction();
}

function findNearestEnemy(tankId: number): { id: number; x: number; y: number; dist: number } | null {
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    let closestEnemy = null;
    let minDist = Number.MAX_VALUE;

    // Перебираем всех врагов
    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
        const enemyId = TankInputTensor.enemiesData.get(tankId, j * 5);
        const enemyX = TankInputTensor.enemiesData.get(tankId, j * 5 + 1);
        const enemyY = TankInputTensor.enemiesData.get(tankId, j * 5 + 2);

        if (enemyId === 0) continue;

        const dist = Math.hypot(tankX - enemyX, tankY - enemyY);

        if (dist < minDist) {
            minDist = dist;
            closestEnemy = { id: enemyId, x: enemyX, y: enemyY, dist };
        }
    }

    return closestEnemy;
}

function findNearestBullet(tankId: number): {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    dist: number;
    danger: number;
} | null {
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    let closestBullet = null;
    let maxDanger = 0;

    // Перебираем все пули
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        const bulletId = TankInputTensor.bulletsData.get(tankId, i * 5);
        const bulletX = TankInputTensor.bulletsData.get(tankId, i * 5 + 1);
        const bulletY = TankInputTensor.bulletsData.get(tankId, i * 5 + 2);
        const bulletVx = TankInputTensor.bulletsData.get(tankId, i * 5 + 3);
        const bulletVy = TankInputTensor.bulletsData.get(tankId, i * 5 + 4);

        if (bulletId === 0) continue;

        // Анализируем опасность пули
        const bulletSpeed = Math.hypot(bulletVx, bulletVy);
        if (bulletSpeed < 0.001) continue;

        const bulletDirX = bulletVx / bulletSpeed;
        const bulletDirY = bulletVy / bulletSpeed;

        // Вектор от пули к танку
        const toTankX = tankX - bulletX;
        const toTankY = tankY - bulletY;

        // Определяем, движется ли пуля к танку
        const dotProduct = toTankX * bulletDirX + toTankY * bulletDirY;

        // Если пуля движется к танку
        if (dotProduct > 0) {
            // Проекция вектора на направление пули
            const projLength = dotProduct;

            // Точка ближайшего прохождения пули к танку
            const closestPointX = bulletX + bulletDirX * projLength;
            const closestPointY = bulletY + bulletDirY * projLength;

            // Расстояние в точке наибольшего сближения
            const minDist = Math.hypot(closestPointX - tankX, closestPointY - tankY);

            // Оценка опасности пули
            if (minDist < 120) { // Увеличенное расстояние обнаружения
                // Время до точки сближения
                const timeToClosest = projLength / bulletSpeed;

                // Плавная оценка опасности
                if (timeToClosest < 1.2) {
                    // Используем smoothstep для плавного изменения опасности
                    const distanceFactor = smoothstep(120, 40, minDist); // От 0 до 1 при приближении
                    const timeFactor = smoothstep(1.2, 0.1, timeToClosest); // От 0 до 1 при приближении

                    const danger = distanceFactor * timeFactor;

                    // Выбираем пулю с наибольшей опасностью
                    if (danger > maxDanger) {
                        maxDanger = danger;
                        closestBullet = {
                            id: bulletId,
                            x: bulletX,
                            y: bulletY,
                            vx: bulletVx,
                            vy: bulletVy,
                            dist: Math.hypot(tankX - bulletX, tankY - bulletY),
                            danger,
                        };
                    }
                }
            }
        }
    }

    return closestBullet;
}

/**
 * Генерация полностью случайных действий
 * @returns Массив с действием [shoot, move, rotate, aimX, aimY]
 */
export function generatePureRandomAction(): number[] {
    // Стрельба: случайное 0 или 1
    const shootRandom = random() > 0.5 ? 1 : 0;
    // Движение вперед-назад
    const moveRandom = randomSign() * random();
    // Поворот влево-вправо
    const rotateRandom = randomSign() * random();
    // Прицеливание
    const aimXRandom = randomSign() * random();
    const aimYRandom = randomSign() * random();

    return [shootRandom, moveRandom, rotateRandom, aimXRandom, aimYRandom];
}
