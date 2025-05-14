import { GameDI } from '../../../DI/GameDI.ts';
import { hasComponent, query } from 'bitecs';
import { Tank } from '../../Components/Tank.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../src/ECS/Components/Transform.ts';
import { Children } from '../../Components/Children.ts';
import { TankPart } from '../../Components/TankPart.ts';
import { isOutOfGameZone } from '../utils/isOutOfGameZone.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { random } from '../../../../../../lib/random.ts';
import { TANK_APPROXIMATE_COLLIDER_RADIUS } from '../../Components/HeuristicsData.ts';

export function createTankDecayOutOfZoneSystem({ world } = GameDI) {
    return () => {
        const tankEids = query(world, [Tank]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const globalTransform = GlobalTransform.matrix.getBatch(tankEid);
            const x = getMatrixTranslationX(globalTransform);
            const y = getMatrixTranslationY(globalTransform);

            if (isOutOfGameZone(x, y, TANK_APPROXIMATE_COLLIDER_RADIUS / 2) && random() < 0.3) {
                const childrenEids = Children.entitiesIds.getBatch(tankEid);
                const childrenCount = Children.entitiesCount[tankEid];

                for (let j = 0; j < childrenCount; j++) {
                    const childEid = childrenEids[j];
                    const isTankPart = hasComponent(world, childEid, TankPart);
                    const isNotParent = !hasComponent(world, childEid, Children);

                    // TODO: we can check part recursively
                    if (isTankPart && isNotParent && random() < 0.05) {
                        scheduleRemoveEntity(childEid);
                        break;
                    }
                }
            }
        }
    };
}