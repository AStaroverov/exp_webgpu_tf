import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';
import { addTransformComponents } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ShapeKind } from '../../../../../renderer/src/ECS/Components/Shape.ts';
import { createRectangleRigidGroup } from '../Components/RigidGroup.ts';
import { HeadlightConfig } from '../../Config/index.ts';
import { PI } from '../../../../../../lib/math.ts';
import { type TankOptions } from './Tank/Common/Options.ts';

export interface TurretHeadlightOptions {
    startX: number;
    startY: number;
    length: number;
    nearWidth: number;
    farWidth: number;
}

export function createTurretHeadlight(
    turretEid: EntityId,
    turretPid: number,
    beam: TurretHeadlightOptions,
    options: TankOptions,
    { world, physicalWorld } = GameDI,
) {
    const { Joint, Parent, Children, Shape, Color, LightEmitter } = getGameComponents(world);

    const [eid, pid] = createRectangleRigidGroup({
        ...options,
        width: 1,
        height: 1,
        density: options.density * 0.001,
        belongsCollisionGroup: 0,
        interactsCollisionGroup: 0,
    });

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(
            new Vector2(beam.startX + beam.length / 2, beam.startY), -PI / 2,
            new Vector2(0, 0), 0,
        ),
        physicalWorld.getRigidBody(turretPid),
        physicalWorld.getRigidBody(pid),
        false,
    );
    Joint.addComponent(world, eid, joint.handle);

    Shape.addComponent(world, eid, ShapeKind.Trapezoid, beam.nearWidth, beam.length, beam.farWidth);
    Color.addComponent(world, eid, ...HeadlightConfig.color);
    LightEmitter.addComponent(world, eid,
        HeadlightConfig.directional ? -HeadlightConfig.intensity : HeadlightConfig.intensity);

    addTransformComponents(world, eid);
    Parent.addComponent(world, eid, turretEid);
    Children.addComponent(world, eid);
    Children.addChildren(turretEid, eid);

    return eid;
}
