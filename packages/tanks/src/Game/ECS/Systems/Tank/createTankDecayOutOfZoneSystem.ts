import { GameDI } from '../../../DI/GameDI.ts';
import { query } from 'bitecs';
import { Vehicle } from '../../Components/Vehicle.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { Children } from '../../Components/Children.ts';
import { isOutOfGameZone } from '../utils/isOutOfGameZone.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { random } from '../../../../../../../lib/random.ts';
import { getSlotFillerEid, isSlot, isSlotEmpty } from '../../Utils/SlotUtils.ts';

// Distance beyond game zone boundary where decay starts
const DECAY_DISTANCE_THRESHOLD = 400;

export function createTankDecayOutOfZoneSystem({ world } = GameDI) {
    return () => {
        const vehicleEids = query(world, [Vehicle]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const globalTransform = GlobalTransform.matrix.getBatch(vehicleEid);
            const x = getMatrixTranslationX(globalTransform);
            const y = getMatrixTranslationY(globalTransform);
            
            if (isOutOfGameZone(x, y, DECAY_DISTANCE_THRESHOLD) && random() < 0.3) {
                const childrenEids = Children.entitiesIds.getBatch(vehicleEid);
                const childrenCount = Children.entitiesCount[vehicleEid];

                for (let j = 0; j < childrenCount; j++) {
                    const slotEid = childrenEids[j];

                    if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;

                    const partEid = getSlotFillerEid(slotEid);
                    if (partEid === 0) continue;

                    if (random() < 0.05) {
                        scheduleRemoveEntity(partEid);
                        break;
                    }
                }
            }
        }
    };
}