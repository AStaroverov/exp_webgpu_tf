import { Worlds } from '../../../DI/Worlds.ts';
import { hasComponent, query } from 'bitecs';
import { destroyTank, getTankHealth } from '../../Entities/Tank/TankUtils.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';

export function createTankAliveSystem({ physicsWorld, renderWorld } = Worlds) {
    const { Vehicle } = getPhysicsWorldComponents(physicsWorld);

    return () => {
        const { Children } = getRenderWorldComponents(renderWorld);
        const vehicleEids = query(physicsWorld, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            const vehicleRenderEid = BridgeDI.getRenderOf(vehicleEid);
            if (!hasComponent(renderWorld, vehicleRenderEid, Children)) continue;

            const hp = getTankHealth(vehicleEid);
            hp === 0 && destroyTank(vehicleEid);
        }
    };
}
