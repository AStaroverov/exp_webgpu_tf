import { addComponent, removeComponent, World } from 'bitecs';
import { delegate } from '../../../../renderer/src/delegate.ts';
import { defineComponent } from '../../../../renderer/src/ECS/utils.ts';

// On a PhysicsWorld atom: physics eid -> render mirror eid.
export const createRenderRefComponent = defineComponent((RenderRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[physicsEid] = renderEid
    return {
        id,
        ref: RenderRef,
        set(world: World, physEid: number, renderEid: number) {
            addComponent(world, physEid, RenderRef);
            id[physEid] = renderEid;
        },
        clear(world: World, physEid: number) {
            id[physEid] = 0;
            removeComponent(world, physEid, RenderRef);
        },
    };
});

// On a RenderWorld mirror: render eid -> physics atom eid.
export const createPhysicsRefComponent = defineComponent((PhysicsRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[renderEid] = physicsEid
    return {
        id,
        ref: PhysicsRef,
        set(world: World, renderEid: number, physEid: number) {
            addComponent(world, renderEid, PhysicsRef);
            id[renderEid] = physEid;
        },
        clear(world: World, renderEid: number) {
            id[renderEid] = 0;
            removeComponent(world, renderEid, PhysicsRef);
        },
    };
});
