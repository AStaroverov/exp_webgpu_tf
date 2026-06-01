import { createWorld, World } from 'bitecs';
import { Opaque } from '../../../../renderer/src/type.ts';

import { createBulletComponent } from './Components/Bullet.ts';
import { createTeamRefComponent } from './Components/TeamRef.ts';
import { createPlayerRefComponent } from './Components/PlayerRef.ts';
import { createDamagableComponent } from './Components/Damagable.ts';
import {
    createDestroyComponent,
    createDestroyBySpeedComponent,
    createDestroyByTimeoutComponent,
} from './Components/Destroy.ts';
import { createHitableComponent } from './Components/Hitable.ts';
import {
    createImpulseComponent,
    createImpulseAtPointComponent,
    createTorqueImpulseComponent,
} from './Components/Impulse.ts';
import { createJointComponent } from './Components/Joint.ts';
import { createJointMotorComponent } from './Components/JointMotor.ts';
import { createObstacleComponent } from './Components/Obstacle.ts';
import {
    createRigidBodyRefComponent,
    createRigidBodyStateComponent,
} from './Components/Physical.ts';
import { createProgressComponent } from './Components/Progress.ts';
import { createTrackComponent } from './Components/Track.ts';
import {
    createVehiclePartComponent,
    createVehiclePartCaterpillarComponent,
} from './Components/VehiclePart.ts';
import { createVehicleTurretComponent } from './Components/VehicleTurret.ts';
import {
    createWheelComponent,
    createWheelDriveComponent,
    createWheelSteerableComponent,
} from './Components/Wheel.ts';

function createPhysicsOnlyComponents(world: World) {
    return {
        // physics bridge
        RigidBodyRef: createRigidBodyRefComponent(world),
        RigidBodyState: createRigidBodyStateComponent(world),
        Impulse: createImpulseComponent(world),
        ImpulseAtPoint: createImpulseAtPointComponent(world),
        TorqueImpulse: createTorqueImpulseComponent(world),
        Joint: createJointComponent(world),
        JointMotor: createJointMotorComponent(world),
        // per-atom gameplay on the contact path
        Hitable: createHitableComponent(world),
        Damagable: createDamagableComponent(world),
        // cheap STATIC team/player copy on the atom (parts + bullets) for the contact
        // fast-path; canonical TeamRef/PlayerRef live on the brain (BrainWorld).
        TeamRef: createTeamRefComponent(world),
        PlayerRef: createPlayerRefComponent(world),
        VehiclePart: createVehiclePartComponent(world),
        VehiclePartCaterpillar: createVehiclePartCaterpillarComponent(world),
        VehicleTurret: createVehicleTurretComponent(world),
        Wheel: createWheelComponent(world),
        WheelDrive: createWheelDriveComponent(world),
        WheelSteerable: createWheelSteerableComponent(world),
        Track: createTrackComponent(world),
        Bullet: createBulletComponent(world),
        Obstacle: createObstacleComponent(world),
        // lifecycle on the atom
        Destroy: createDestroyComponent(world),
        DestroyBySpeed: createDestroyBySpeedComponent(world),
        DestroyByTimeout: createDestroyByTimeoutComponent(world),
        Progress: createProgressComponent(world),
    };
}

export type PhysicsWorldComponents = ReturnType<typeof createPhysicsOnlyComponents>;

export type PhysicsWorld = Opaque<'PhysicsWorld', World<{
    components: PhysicsWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createPhysicsWorld(): PhysicsWorld {
    const context = {
        components: null as unknown as PhysicsWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as PhysicsWorld;
    context.components = createPhysicsOnlyComponents(world);
    return world;
}

export function getPhysicsWorldComponents(world: PhysicsWorld): PhysicsWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Physics components are not available on this world');
    }
    return components;
}
