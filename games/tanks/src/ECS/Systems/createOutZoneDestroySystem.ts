import { DI } from '../../DI';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { Not, query } from 'bitecs';
import { Parent } from '../Components/Parent.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createOutZoneDestroySystem({ world, canvas } = DI) {
    return () => {
        const { width, height } = canvas;
        const eids = query(world, [GlobalTransform, Not(Parent)]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const globalTransform = GlobalTransform.matrix.getBatche(eid);
            const x = getMatrixTranslationX(globalTransform);
            const y = getMatrixTranslationY(globalTransform);

            if (x < -1000 || x > width + 1000 || y < -1000 || y > height + 1000) {
                scheduleRemoveEntity(eid);
            }
        }
    };
}
