import { Children } from '../../../games/tanks/src/ECS/Components/Children.ts';
import { GlobalTransform, LocalTransform } from '../Components/Transform.ts';
import { mat4 } from 'gl-matrix';
import { World } from '../world.ts';
import { query } from 'bitecs';

export function createTransformSystem(world: World) {
    return function execMainTransformSystem() {
        {
            const entities = query(world, [LocalTransform, GlobalTransform]);

            for (let i = 0; i < entities.length; i++) {
                const id = entities[i];
                const local = LocalTransform.matrix.getBatch(id);
                const global = GlobalTransform.matrix.getBatch(id);
                mat4.copy(global, local);
            }
        }

        {
            const entities = query(world, [GlobalTransform, Children]);

            for (let i = 0; i < entities.length; i++) {
                const id = entities[i];
                const globalParent = GlobalTransform.matrix.getBatch(id);
                for (let j = 0; j < Children.entitiesCount[id]; j++) {
                    const childId = Children.entitiesIds.get(id, j);
                    const localChild = LocalTransform.matrix.getBatch(childId);
                    const globalChild = GlobalTransform.matrix.getBatch(childId);
                    mat4.multiply(globalChild, globalParent, localChild);
                }
            }
        }
    };
}