import { mat4 } from 'gl-matrix';
import { getRenderComponents, type RenderWorldLike } from '../world.ts';
import { query } from 'bitecs';
type ChildrenLike = {
    entitiesCount: ArrayLike<number>;
    entitiesIds: { get(eid: number, i: number): number };
};

export function createTransformSystem(world: RenderWorldLike, Children: ChildrenLike) {
    const { GlobalTransform, LocalTransform } = getRenderComponents(world);
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
            const entities = query(world, [GlobalTransform, Children as object]);

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
