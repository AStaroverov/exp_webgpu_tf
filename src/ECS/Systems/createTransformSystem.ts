import { Children } from '../../../games/tanks/src/ECS/Components/Children.ts';
import { defineQuery } from 'bitecs';
import { GlobalTransform, LocalTransform } from '../Components/Transform.ts';
import { mat4 } from 'gl-matrix';
import { World } from '../world.ts';

export function createTransformSystem(world: World) {
    const queryAll = defineQuery([LocalTransform, GlobalTransform]);
    const queryGlobalWithChilds = defineQuery([GlobalTransform, Children]);

    return function execMainTransformSystem() {
        {
            const entities = queryAll(world);

            for (let i = 0; i < entities.length; i++) {
                const id = entities[i];
                const local = LocalTransform.matrix[id];
                const global = GlobalTransform.matrix[id];
                mat4.copy(global, local);
            }
        }

        {
            const entities = queryGlobalWithChilds(world);

            for (let i = 0; i < entities.length; i++) {
                const id = entities[i];
                const globalParent = GlobalTransform.matrix[id];
                for (let j = 0; j < Children.entitiesCount[id]; j++) {
                    const childId = Children.entitiesIds[id][j];
                    const localChild = LocalTransform.matrix[childId];
                    const globalChild = GlobalTransform.matrix[childId];
                    mat4.multiply(globalChild, globalParent, localChild);
                }
            }
        }
    };
}