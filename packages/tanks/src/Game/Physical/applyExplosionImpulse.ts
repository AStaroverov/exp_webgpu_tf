import { EntityId } from "bitecs";
import { GlobalTransform, getMatrixTranslationX, getMatrixTranslationY } from "renderer/src/ECS/Components/Transform";
import { Impulse, TorqueImpulse } from "../ECS/Components/Impulse";

// Explosion force settings
const EXPLOSION_IMPULSE_BASE = 1000000;      // Base impulse strength
const EXPLOSION_IMPULSE_RANDOM = EXPLOSION_IMPULSE_BASE / 2;    // Random additional impulse
const EXPLOSION_TORQUE_BASE = 10000000;        // Base angular impulse
const EXPLOSION_TORQUE_RANDOM = EXPLOSION_TORQUE_BASE / 2;     // Random additional torque
const EXPLOSION_UPWARD_BIAS = 0.3;       // Slight upward bias for more dramatic effect

export function applyExplosionImpulse(
    eid: EntityId,
    explosionX: number,
    explosionY: number,
) {
    // Get part position
    const partMatrix = GlobalTransform.matrix.getBatch(eid);
    const partX = getMatrixTranslationX(partMatrix);
    const partY = getMatrixTranslationY(partMatrix);

    // Calculate direction from explosion center to part
    let dirX = partX - explosionX;
    let dirY = partY - explosionY;

    // Normalize direction
    const dist = Math.sqrt(dirX * dirX + dirY * dirY);
    if (dist > 0.01) {
        dirX /= dist;
        dirY /= dist;
    } else {
        // If part is at explosion center, use random direction
        const angle = Math.random() * Math.PI * 2;
        dirX = Math.cos(angle);
        dirY = Math.sin(angle);
    }

    // Add some randomness to direction
    const spreadAngle = (Math.random() - 0.5) * 0.8; // ±0.4 radians spread
    const cosSpread = Math.cos(spreadAngle);
    const sinSpread = Math.sin(spreadAngle);
    const newDirX = dirX * cosSpread - dirY * sinSpread;
    const newDirY = dirX * sinSpread + dirY * cosSpread;
    dirX = newDirX;
    dirY = newDirY;

    // Add slight upward bias (negative Y in screen coords)
    dirY -= EXPLOSION_UPWARD_BIAS;

    // Calculate impulse with randomness
    const impulseStrength = EXPLOSION_IMPULSE_BASE + Math.random() * EXPLOSION_IMPULSE_RANDOM;
    const impulseX = dirX * impulseStrength;
    const impulseY = dirY * impulseStrength;

    // Add linear impulse to component (will be applied by system)
    Impulse.add(eid, impulseX, impulseY);

    // Add random angular impulse for spinning effect
    const torque = (Math.random() - 0.5) * 2 * (EXPLOSION_TORQUE_BASE + Math.random() * EXPLOSION_TORQUE_RANDOM);
    TorqueImpulse.add(eid, torque);
}
