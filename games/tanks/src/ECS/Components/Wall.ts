import { createRectangleRR } from './RigidRender.ts';
import { DI } from '../../DI';
import { addComponent, defineComponent } from 'bitecs';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addHitableComponent } from './Hitable.ts';

export const Wall = defineComponent();

type Options = Parameters<typeof createRectangleRR>[0]

const mutatedOptions: Options = {
    x: 0,
    y: 0,
    width: 10,
    height: 10,
    rotation: 0,
    color: new Float32Array([0.9, 0.9, 0.9, 1]),
    shadow: new Float32Array([0, 2]),
    mass: 1000,
    angularDamping: 0.7,
    linearDamping: 0.7,
    // collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
    belongsCollisionGroup: CollisionGroup.WALL,
};
const defaultOptions = structuredClone(mutatedOptions);
const resetOptions = (options: Partial<Options>) => {
    mutatedOptions.x = options.x ?? defaultOptions.x;
    mutatedOptions.y = options.y ?? defaultOptions.y;
    mutatedOptions.width = options.width ?? defaultOptions.width;
    mutatedOptions.height = options.height ?? defaultOptions.height;
    mutatedOptions.rotation = options.rotation ?? defaultOptions.rotation;
    (mutatedOptions.color as Float32Array).set(options.color ?? defaultOptions.color);
    (mutatedOptions.shadow as Float32Array).set(options.shadow ?? defaultOptions.shadow);
    mutatedOptions.mass = options.mass ?? defaultOptions.mass;
    mutatedOptions.angularDamping = options.angularDamping ?? defaultOptions.angularDamping;
    mutatedOptions.linearDamping = options.linearDamping ?? defaultOptions.linearDamping;
    // mutatedOptions.collisionEvent = defaultOptions.collisionEvent;
    mutatedOptions.belongsCollisionGroup = defaultOptions.belongsCollisionGroup;
};

export function createWallRR(options: Partial<Options>, { world } = DI) {
    resetOptions(options);
    const [eid] = createRectangleRR(mutatedOptions);

    addComponent(world, Wall, eid);
    addHitableComponent(world, eid);
    return eid;
}
