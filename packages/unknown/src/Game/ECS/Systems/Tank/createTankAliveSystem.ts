import { Worlds } from '../../../DI/Worlds.ts';
import { query } from 'bitecs';
import { destroyTank, getTankHealth } from '../../Entities/Tank/TankUtils.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeChildren, getNodePhysics } from '../../refs.ts';

export function createTankAliveSystem({ brainWorld } = Worlds) {
    const { Vehicle } = getBrainWorldComponents(brainWorld);

    return () => {
        const brainEids = query(brainWorld, [Vehicle]);

        for (const brainEid of brainEids) {
            // brainEid IS the hull node; its presentation (downward) is the hull atom.
            const vehicleEid = getNodePhysics(brainEid);
            // skip until it has Brain children (fully built).
            if (getNodeChildren(brainEid).length === 0) continue;

            const hp = getTankHealth(vehicleEid);
            hp === 0 && destroyTank(vehicleEid);
        }
    };
}
