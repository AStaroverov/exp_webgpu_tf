import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { Children } from '../../Components/Children.ts';
import { query } from 'bitecs';
import { destroyTank, getTankHealth } from '../../Entities/Tank/TankUtils.ts';

export function createTankAliveSystem({ world } = GameDI) {
    return () => {
        const tankEids = query(world, [Tank, Children]);

        for (const tankEid of tankEids) {
            const hp = getTankHealth(tankEid);
            hp === 0 && destroyTank(tankEid);
        }
    };
}
