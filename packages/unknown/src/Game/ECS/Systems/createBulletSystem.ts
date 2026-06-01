import { query } from 'bitecs';
import { Worlds } from '../../DI/Worlds.ts';
import { getNodeParent } from '../refs.ts';
import { spawnBullet } from '../Entities/Bullet.ts';
import { getBrainWorldComponents } from '../createBrainWorld.ts';

export function createSpawnerBulletsSystem({ brainWorld } = Worlds) {
    const { Firearms, TurretController } = getBrainWorldComponents(brainWorld);

    return ((delta: number) => {
        // Node-rooted: iterate armed turret NODES directly (Firearms + TurretController).
        // This is exactly the old set — only armed turrets ever carried both — without
        // the atom->brain climb (harvester barriers lack Firearms, so are excluded).
        const turretBrains = query(brainWorld, [Firearms, TurretController]);

        for (let i = 0; i < turretBrains.length; i++) {
            const turretBrain = turretBrains[i];

            Firearms.updateReloading(turretBrain, delta);
            if (!TurretController.shouldShoot(turretBrain) || Firearms.isReloading(turretBrain)) continue;
            Firearms.startReloading(turretBrain);

            // turret node -> Brain parent (hull node) = the hull-brain.
            spawnBullet(getNodeParent(turretBrain));
        }
    });
}
