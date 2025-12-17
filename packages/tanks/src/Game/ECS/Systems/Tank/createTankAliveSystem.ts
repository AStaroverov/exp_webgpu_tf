import { GameDI } from '../../../DI/GameDI.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { Children } from '../../Components/Children.ts';
import { query } from 'bitecs';
import { destroyTank, getTankHealth } from '../../Entities/Tank/TankUtils.ts';

export function createTankAliveSystem({ world } = GameDI) {
    return () => {
        const vehicleEids = query(world, [Vehicle, Children]);

        for (const vehicleEid of vehicleEids) {
            const hp = getTankHealth(vehicleEid);
            hp === 0 && destroyTank(vehicleEid);
        }
    };
}
