import { GlobalTransform, setMatrixRotateZ, setMatrixTranslate } from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef, RigidBodyState } from '../Components/Physical.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { query } from 'bitecs';

export function createApplyRigidBodyToTransformSystem({ world } = GameDI) {
    return function () {
        const entities = query(world, [GlobalTransform, RigidBodyRef]);

        for (let i = 0; i < entities.length; i++) {
            const eid = entities[i];
            const globalMatrix = GlobalTransform.matrix.getBatche(eid);
            setMatrixTranslate(globalMatrix, RigidBodyState.position.get(eid, 0), RigidBodyState.position.get(eid, 1));
            setMatrixRotateZ(globalMatrix, RigidBodyState.rotation[eid]);
        }
    };
}

