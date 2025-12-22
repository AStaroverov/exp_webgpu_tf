import { ActiveCollisionTypes, ActiveEvents, RigidBodyType } from '@dimforge/rapier2d-simd';
import { EntityId } from 'bitecs';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { SpiceCollectorPhysicsConfig } from '../../../Config/index.ts';
import { createCircleRR } from '../../Components/RigidRender.ts';
import { SpiceCollector } from '../../Components/SpiceCollector.ts';
import { TeamRef } from '../../Components/TeamRef.ts';

/** Default radius for spice collector */
const DEFAULT_RADIUS = 300;

/** Default color - semi-transparent khaki color */
const DEFAULT_COLOR: TColor = new Float32Array([189/255, 183/255, 108/255, 1]);

/**
 * Options for creating a spice collector
 */
export type CreateSpiceCollectorOptions = {
    x: number;
    y: number;
    teamId: number;
    /** Radius of the collector (both visual and trigger) */
    radius?: number;
    /** Color of the collector */
    color?: TColor;
};

/**
 * Create a spice collector entity.
 * A sensor circle that detects spice collision and triggers collection.
 */
export function createSpiceCollector(
    opts: CreateSpiceCollectorOptions,
    { world } = GameDI
): EntityId {
    const radius = opts.radius ?? DEFAULT_RADIUS;
    const color = opts.color ?? DEFAULT_COLOR;

    const rrOptions = {
        x: opts.x,
        y: opts.y,
        z: SpiceCollectorPhysicsConfig.z,
        radius,
        color,
        bodyType: RigidBodyType.Fixed,
        density: 0,
        // Sensor - detects intersections but no physics response
        sensor: true,
        // Enable collision events for intersection detection
        collisionEvent: ActiveEvents.COLLISION_EVENTS,
        activeCollisionTypes: ActiveCollisionTypes.DEFAULT | ActiveCollisionTypes.KINEMATIC_FIXED,
        // Collision groups
        belongsCollisionGroup: SpiceCollectorPhysicsConfig.belongsCollisionGroup,
        interactsCollisionGroup: SpiceCollectorPhysicsConfig.interactsCollisionGroup,
        belongsSolverGroup: SpiceCollectorPhysicsConfig.belongsSolverGroup,
        interactsSolverGroup: SpiceCollectorPhysicsConfig.interactsSolverGroup,
    };

    const [collectorEid] = createCircleRR(rrOptions);

    // Add components
    SpiceCollector.addComponent(world, collectorEid);
    TeamRef.addComponent(world, collectorEid, opts.teamId);

    return collectorEid;
}

