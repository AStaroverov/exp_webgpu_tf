import { DI } from '../../DI';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { defineQuery, Not } from 'bitecs';
import { Parent } from '../Components/Parent.ts';
import { recursiveTypicalRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createOutZoneDestroySystem({ world, canvas } = DI) {
    const query = defineQuery([GlobalTransform, Not(Parent)]);

    return () => {
        const { width, height } = canvas;
        const eids = query(world);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const globalTransform = GlobalTransform.matrix[eid];
            const x = getMatrixTranslationX(globalTransform);
            const y = getMatrixTranslationY(globalTransform);

            if (x < -100 || x > width + 100 || y < -100 || y > height + 100) {
                recursiveTypicalRemoveEntity(eid);
            }
        }
    };
}
