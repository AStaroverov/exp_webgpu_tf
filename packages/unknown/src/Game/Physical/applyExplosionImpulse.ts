import { EntityId } from 'bitecs';
import { GlobalTransform, getMatrixTranslationX, getMatrixTranslationY } from 'renderer/src/ECS/Components/Transform';
import { GameDI } from '../DI/GameDI.ts';
import { getGameComponents } from '../ECS/createGameWorld.ts';

const EXPLOSION_IMPULSE_BASE = 1000000;
const EXPLOSION_IMPULSE_RANDOM = EXPLOSION_IMPULSE_BASE / 2;
const EXPLOSION_TORQUE_BASE = 10000000;
const EXPLOSION_TORQUE_RANDOM = EXPLOSION_TORQUE_BASE / 2;
const EXPLOSION_UPWARD_BIAS = 0.3;

export function applyExplosionImpulse(
    eid: EntityId,
    explosionX: number,
    explosionY: number,
    { world } = GameDI,
) {
    const { Impulse, TorqueImpulse } = getGameComponents(world);

    const partMatrix = GlobalTransform.matrix.getBatch(eid);
    const partX = getMatrixTranslationX(partMatrix);
    const partY = getMatrixTranslationY(partMatrix);

    let dirX = partX - explosionX;
    let dirY = partY - explosionY;

    const dist = Math.sqrt(dirX * dirX + dirY * dirY);
    if (dist > 0.01) {
        dirX /= dist;
        dirY /= dist;
    } else {
        const angle = Math.random() * Math.PI * 2;
        dirX = Math.cos(angle);
        dirY = Math.sin(angle);
    }

    const spreadAngle = (Math.random() - 0.5) * 0.8;
    const cosSpread = Math.cos(spreadAngle);
    const sinSpread = Math.sin(spreadAngle);
    const newDirX = dirX * cosSpread - dirY * sinSpread;
    const newDirY = dirX * sinSpread + dirY * cosSpread;
    dirX = newDirX;
    dirY = newDirY;

    dirY -= EXPLOSION_UPWARD_BIAS;

    const impulseStrength = EXPLOSION_IMPULSE_BASE + Math.random() * EXPLOSION_IMPULSE_RANDOM;
    const impulseX = dirX * impulseStrength;
    const impulseY = dirY * impulseStrength;

    Impulse.add(eid, impulseX, impulseY);

    const torque = (Math.random() - 0.5) * 2 * (EXPLOSION_TORQUE_BASE + Math.random() * EXPLOSION_TORQUE_RANDOM);
    TorqueImpulse.add(eid, torque);
}
