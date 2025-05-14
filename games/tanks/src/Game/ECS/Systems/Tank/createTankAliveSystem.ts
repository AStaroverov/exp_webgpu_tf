import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { Children } from '../../Components/Children.ts';
import { query } from 'bitecs';
import { getTankHealth, removeTankComponentsWithoutParts, tearOffTankPart } from '../../Entities/Tank/TankUtils.ts';

export function createTankAliveSystem({ world } = GameDI) {
    return () => {
        const tankEids = query(world, [Tank, Children]);

        for (const tankEid of tankEids) {
            const hp = getTankHealth(tankEid);

            if (hp === 0) {
                // turret
                const turretEid = Tank.turretEId[tankEid];
                for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
                    const eid = Children.entitiesIds.get(turretEid, i);
                    tearOffTankPart(eid, false);
                }
                Children.removeAllChildren(turretEid);
                // tank parts
                for (let i = 0; i < Children.entitiesCount[tankEid]; i++) {
                    const partEid = Children.entitiesIds.get(tankEid, i);
                    tearOffTankPart(partEid, false);
                }
                Children.removeAllChildren(tankEid);
                removeTankComponentsWithoutParts(tankEid);
            }
        }
    };
}
