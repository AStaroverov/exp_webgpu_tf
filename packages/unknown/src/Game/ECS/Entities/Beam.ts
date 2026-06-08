/**
 * Beam — the render-only light cone the Ranger projects, plus the queries over it.
 * It is the concrete thing "beam" refers to everywhere: a child entity of the turret
 * (no rigid body — the attached transform system keeps `global = turretGlobal * local`
 * in sync each tick), stored in `BeamRef.beamEid`.
 *
 * This module owns the beam's whole behaviour:
 *   - `createBeam`               — build the entity (trapezoid + light emitter).
 *   - `activateBeam` / `deactivateBeam` — the pulse trigger and end (drive the owner's
 *     `Progress` clock + the beam's visibility).
 *   - `isBeamActive`             — is the pulse still live this tick.
 *   - `getBeamTargets`           — what the beam physically lights this tick.
 */

import { addEntity, EntityId, hasComponent, removeComponent } from 'bitecs';
import { Ball, Capsule, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../DI/GameDI.ts';
import { MapDI } from '../../DI/MapDI.ts';
import { getGameComponents } from '../createGameWorld.ts';
import { getEntityIdByPhysicalId } from '../Components/Physical.ts';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';
import { HexGridConfig } from '../../Map/HexConfig.ts';
import {
    addTransformComponents,
    applyMatrixRotateZ,
    applyMatrixTranslate,
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ShapeKind } from '../../../../../renderer/src/ECS/Components/Shape.ts';
import { HeadlightConfig, SpotlightConfig, SpottingConfig, ZIndexConfig } from '../../Config/index.ts';
import { PI } from '../../../../../../lib/math.ts';
import { revealByFire } from './Vehicle/VehicleBase.ts';

/** Beam z: above the hull parts (TankHull), below the turret head (TankTurret). */
export interface BeamOptions {
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
export function createBeam(
    turretEid: EntityId,
    beam: BeamOptions,
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

export function activateBeam(ownerEid: EntityId, { world } = GameDI): void {
    const { Progress, BeamRef } = getGameComponents(world);
    Progress.addComponent(world, ownerEid, SpotlightConfig.pulseMs); // (re)start the pulse
    setBeamVisible(BeamRef.getBeamEid(ownerEid), true);
    revealByFire(ownerEid);
}

export function deactivateBeam(ownerEid: EntityId, { world } = GameDI): void {
    const { BeamRef } = getGameComponents(world);
    setBeamVisible(BeamRef.getBeamEid(ownerEid), false);
}

export function isBeamActive(ownerEid: EntityId, { world } = GameDI): boolean {
    const { Progress } = getGameComponents(world);
    return Progress.getProgress(ownerEid) < 1;
}

const BEAM_RADIUS = HexGridConfig.radius * 1.3;
const BEAM_WORLD_REACH = SpottingConfig.beamLength * Math.sqrt(3) * HexGridConfig.radius;
const OBSTACLE_GROUPS = createCollisionGroups(CollisionGroup.ALL, CollisionGroup.OBSTACLE);
const VEHICLE_GROUPS = createCollisionGroups(CollisionGroup.ALL, CollisionGroup.VEHICALE_BASE);

// Reused query shapes/vectors — no per-call alloc.
const shapePos = new Vector2(0, 0);
const shapeVel = new Vector2(0, 0);
const ball = new Ball(BEAM_RADIUS);
const capsule = new Capsule(1, BEAM_RADIUS); // halfHeight is set per call

export interface BeamCellSink {
    add(q: number, r: number): void;
}
export function getBeamTargets(
    rangerEid: number,
    out: Set<number>,
    cells?: BeamCellSink,
    { world, physicalWorld } = GameDI,
): void {
    const { Tank } = getGameComponents(world);

    const turretEid = Tank.turretEId[rangerEid];
    const matrix = GlobalTransform.matrix.getBatch(turretEid);
    const angle = getMatrixRotationZ(matrix);
    const dirX = Math.cos(angle);
    const dirY = Math.sin(angle);
    const startX = getMatrixTranslationX(matrix);
    const startY = getMatrixTranslationY(matrix);

    // 1) Cut distance: sweep the circle along the beam against obstacles.
    shapePos.x = startX;
    shapePos.y = startY;
    shapeVel.x = dirX;
    shapeVel.y = dirY;
    const hit = physicalWorld.castShape(
        shapePos, 0, shapeVel, ball,
        0, BEAM_WORLD_REACH, true,
        undefined, OBSTACLE_GROUPS,
    );
    const reach = hit !== null ? hit.time_of_impact : BEAM_WORLD_REACH;
    if (reach <= 0) return; // beam start is inside an obstacle

    // 2) Everything under the swept circle: a capsule over [0, reach]. The
    //    capsule's axis is local Y, so rotate by `angle − π/2` to align it.
    capsule.halfHeight = reach / 2;
    shapePos.x = startX + dirX * (reach / 2);
    shapePos.y = startY + dirY * (reach / 2);
    physicalWorld.intersectionsWithShape(
        shapePos, angle - Math.PI / 2, capsule,
        (collider) => {
            const body = collider.parent();
            if (body !== null) out.add(getEntityIdByPhysicalId(body.handle));
            return true; // keep collecting
        },
        undefined, VEHICLE_GROUPS,
    );

    if (cells) rasterizeBeamCells(startX, startY, dirX, dirY, reach, cells);
}

function rasterizeBeamCells(
    startX: number,
    startY: number,
    dirX: number,
    dirY: number,
    reach: number,
    cells: BeamCellSink,
): void {
    const grid = MapDI.grid;
    if (!grid) return;

    const radius2 = BEAM_RADIUS * BEAM_RADIUS;
    const stride = HexGridConfig.radius * 0.5; // below the inradius — no cell skipped
    let lastQ = NaN;
    let lastR = NaN;
    for (let t = 0; t <= reach; t += stride) {
        const sx = startX + dirX * t;
        const sy = startY + dirY * t;
        const sampleHex = grid.worldToHex(sx, sy);
        if (!sampleHex) continue;
        // Re-testing the same sample cell across consecutive samples is cheap to
        // skip; the neighbour fan still covers the wider-than-cell beam radius.
        if (sampleHex.q === lastQ && sampleHex.r === lastR) continue;
        lastQ = sampleHex.q;
        lastR = sampleHex.r;

        markIfCovered(grid, sampleHex.q, sampleHex.r, startX, startY, dirX, dirY, reach, radius2, cells);
        const neighbours = grid.neighbors(sampleHex);
        for (let i = 0; i < neighbours.length; i++) {
            markIfCovered(grid, neighbours[i].q, neighbours[i].r, startX, startY, dirX, dirY, reach, radius2, cells);
        }
    }
}

/** Mark `(q, r)` if its world center is within `BEAM_RADIUS` of the beam segment. */
function markIfCovered(
    grid: NonNullable<typeof MapDI.grid>,
    q: number,
    r: number,
    startX: number,
    startY: number,
    dirX: number,
    dirY: number,
    reach: number,
    radius2: number,
    cells: BeamCellSink,
): void {
    const center = grid.hexToWorld(q, r);
    if (!center) return;
    // Closest point on the segment [0, reach] to the cell center (dir is unit).
    const proj = Math.max(0, Math.min(reach, (center.x - startX) * dirX + (center.y - startY) * dirY));
    const cx = startX + dirX * proj;
    const cy = startY + dirY * proj;
    const ddx = center.x - cx;
    const ddy = center.y - cy;
    if (ddx * ddx + ddy * ddy <= radius2) cells.add(q, r);
}

const stashedShape = new Map<EntityId, { kind: number; values: number[] }>();
function setBeamVisible(eid: EntityId, on: boolean, { world } = GameDI): void {
    if (!eid) return;
    const { Shape } = getGameComponents(world);

    if (on) {
        if (hasComponent(world, eid, Shape)) return; // already shown
        const s = stashedShape.get(eid);
        if (s) {
            const v = s.values;
            Shape.addComponent(world, eid, s.kind, v[0], v[1], v[2], v[3], v[4], v[5]);
        }
        return;
    }

    if (!hasComponent(world, eid, Shape)) return; // already hidden
    stashedShape.set(eid, {
        kind: Shape.kind[eid],
        values: Array.from(Shape.values.getBatch(eid)),
    });
    removeComponent(world, eid, Shape);
}