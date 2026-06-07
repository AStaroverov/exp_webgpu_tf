import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';
import {
    addTransformComponents,
    applyMatrixRotateZ,
    applyMatrixTranslate,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ShapeKind } from '../../../../../renderer/src/ECS/Components/Shape.ts';
import { HeadlightConfig, ZIndexConfig } from '../../Config/index.ts';
import { PI } from '../../../../../../lib/math.ts';

/** Beam z: above the hull parts (TankHull), below the turret head (TankTurret). */
export interface TurretHeadlightOptions {
    startX: number;
    startY: number;
    length: number;
    nearWidth: number;
    farWidth: number;
    light?: {
        color: Float32Array;
        intensity: number;
        directional: boolean;
    };
}

/**
 * Render-only beam attached to the turret: no rigid body — the attached
 * transform system keeps `global = turretGlobal * local` in sync each tick.
 */
export function createTurretHeadlight(
    turretEid: EntityId,
    beam: TurretHeadlightOptions,
    { world } = GameDI,
) {
    const { Parent, Children, Shape, Color, Roundness, Blurness, LightEmitter, LocalTransform } = getGameComponents(world);

    const eid = addEntity(world);

    const light = beam.light ?? HeadlightConfig;
    Shape.addComponent(world, eid, ShapeKind.Trapezoid, beam.nearWidth, beam.length, beam.farWidth);
    Color.addComponent(world, eid, ...light.color);
    Blurness.addComponent(world, eid, 6);
    Roundness.addComponent(world, eid, 10);
    LightEmitter.addComponent(world, eid, light.directional ? -light.intensity : light.intensity);

    addTransformComponents(world, eid);
    const local = LocalTransform.matrix.getBatch(eid);
    applyMatrixTranslate(local, beam.startX + beam.length / 2, beam.startY, ZIndexConfig.TankHull );
    applyMatrixRotateZ(local, -PI / 2);

    Parent.addComponent(world, eid, turretEid);
    Children.addComponent(world, eid);
    Children.addChildren(turretEid, eid);

    return eid;
}
