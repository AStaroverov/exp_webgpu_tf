import { GameDI } from '../../DI/GameDI.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { Not, query } from 'bitecs';
import { Parent } from '../Components/Parent.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createOutZoneDestroySystem({ world, width, height } = GameDI) {
    return () => {
        const eids = query(world, [GlobalTransform, Not(Parent)]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const globalTransform = GlobalTransform.matrix.getBatch(eid);
            const x = getMatrixTranslationX(globalTransform);
            const y = getMatrixTranslationY(globalTransform);

            if (x < -500 || x > width + 500 || y < -500 || y > height + 500) {
                scheduleRemoveEntity(eid);
            }
        }
    };
}
