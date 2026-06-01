import { createWorld, World } from 'bitecs';
import { Opaque } from '../../../../renderer/src/type.ts';

import { createVehicleComponent } from './Components/Vehicle.ts';
import { createVehicleControllerComponent } from './Components/VehicleController.ts';
import { createTurretControllerComponent } from './Components/TurretController.ts';
import { createFirearmsComponent } from './Components/Firearms.ts';
import { createTankComponent } from './Components/Tank.ts';
import { createHeuristicsDataComponent } from './Components/HeuristicsData.ts';
import { createLastHittersComponent } from './Components/LastHitters.ts';
import { createTeamRefComponent } from './Components/TeamRef.ts';
import { createPlayerRefComponent } from './Components/PlayerRef.ts';
import { createParentComponent } from './Components/Parent.ts';
import { createChildrenComponent } from './Components/Children.ts';
import {
    createNodeRenderRefComponent,
    createNodePhysicsRefComponent,
    createNodeSlotsRefComponent,
} from '../DI/links.ts';

// The brain/gameplay world: the "mind" of a vehicle, carved off the physics atoms.
// A tank has TWO brains: a hull-brain (Vehicle/VehicleController/Tank/Heuristics/
// LastHitters + canonical Team/Player) and a turret-brain (TurretController/Firearms).
// Each brain node references its presentation downward (NodeRenderRef/NodePhysicsRef);
// the object tree of nodes lives in Parent/Children here.
function createBrainOnlyComponents(world: World) {
    return {
        Vehicle: createVehicleComponent(world),
        VehicleController: createVehicleControllerComponent(world),
        TurretController: createTurretControllerComponent(world),
        Firearms: createFirearmsComponent(world),
        Tank: createTankComponent(world),
        HeuristicsData: createHeuristicsDataComponent(world),
        LastHitters: createLastHittersComponent(world),
        // canonical team/player (the cheap static copy lives on the atom's own TeamRef/PlayerRef)
        TeamRef: createTeamRefComponent(world),
        PlayerRef: createPlayerRefComponent(world),
        // downward node refs (scheme A: NodeRenderRef XOR NodePhysicsRef, plus NodeSlotsRef)
        NodeRenderRef: createNodeRenderRefComponent(world),
        NodePhysicsRef: createNodePhysicsRefComponent(world),
        NodeSlotsRef: createNodeSlotsRefComponent(world),
        // Brain hierarchy: the object tree of major nodes lives here (turret child of
        // hull, gun child of turret, tracks/wheels children of hull).
        Parent: createParentComponent(world),
        Children: createChildrenComponent(world),
    };
}

export type BrainWorldComponents = ReturnType<typeof createBrainOnlyComponents>;

export type BrainWorld = Opaque<'BrainWorld', World<{
    components: BrainWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createBrainWorld(): BrainWorld {
    const context = {
        components: null as unknown as BrainWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as BrainWorld;
    context.components = createBrainOnlyComponents(world);
    return world;
}

export function getBrainWorldComponents(world: BrainWorld): BrainWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Brain components are not available on this world');
    }
    return components;
}
