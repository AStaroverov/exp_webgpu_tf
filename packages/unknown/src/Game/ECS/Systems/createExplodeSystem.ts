import { GameDI } from '../../DI/GameDI.ts';
import { Not, query } from 'bitecs';
import { getGameComponents } from '../createGameWorld.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { spawnExplosion } from '../Entities/Explosion.ts';
import { spawnLightFlash } from '../Entities/LightFlash.ts';
import { ExplosionConfig, FlashLightConfig } from '../../Config/index.ts';
import { DamageKind } from '../Components/Damagable.ts';

/**
 * Detonates every entity that is both `Explodable` and scheduled for `Destroy`,
 * just before `createDestroySystem` removes it. This keeps the destroy systems
 * dumb (they only add `Destroy`): whatever the reason an explosive entity dies —
 * collision, max range, timeout — it explodes here, uniformly.
 *
 * The blast spawns the VFX/light flash and deals area damage to nearby hitables,
 * scaled by proximity to the epicenter (full at center, zero at the radius edge).
 */
export function createExplodeSystem({ world } = GameDI) {
    const { Explodable, Destroy, Hitable, Bullet } = getGameComponents(world);

    return () => {
        const eids = query(world, [Explodable, Destroy]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const matrix = GlobalTransform.matrix.getBatch(eid);
            const x = getMatrixTranslationX(matrix);
            const y = getMatrixTranslationY(matrix);

            spawnExplosion({
                x,
                y,
                size: Explodable.vfxSize[eid],
                duration: ExplosionConfig.defaultDuration,
            });
            spawnLightFlash({
                x,
                y,
                radius: Explodable.lightRadius[eid],
                duration: FlashLightConfig.explosion.duration,
                color: FlashLightConfig.explosion.color,
                intensity: FlashLightConfig.explosion.intensity,
            });

            const damage = Explodable.damage[eid];
            const radius = Explodable.radius[eid];
            const radiusSq = radius * radius;

            // Damage every hitable (tank parts, rocks, ...) in range, except bullets.
            const targets = query(world, [Hitable, Not(Bullet)]);
            for (let j = 0; j < targets.length; j++) {
                const targetEid = targets[j];
                const targetMatrix = GlobalTransform.matrix.getBatch(targetEid);
                const dx = getMatrixTranslationX(targetMatrix) - x;
                const dy = getMatrixTranslationY(targetMatrix) - y;
                const distSq = dx * dx + dy * dy;
                if (distSq > radiusSq) continue;

                const proximity = 1 - Math.sqrt(distSq) / radius;
                Hitable.hit$(targetEid, eid, damage * proximity, DamageKind.Physical);
            }
        }
    };
}
