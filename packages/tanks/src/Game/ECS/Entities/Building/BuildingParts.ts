import { RigidBodyType } from '@dimforge/rapier2d-simd';
import { EntityId } from 'bitecs';
import { min } from '../../../../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../../../../lib/random.ts';
import { Color, TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { ZIndex } from '../../../consts.ts';
import { Children } from '../../Components/Children.ts';
import { Damagable } from '../../Components/Damagable.ts';
import { Hitable } from '../../Components/Hitable.ts';
import { Parent } from '../../Components/Parent.ts';
import { createRectangleRR } from '../../Components/RigidRender.ts';
import { BuildingPart } from '../../Components/Building.ts';
import { BuildingPartType } from './generateBuildingShape.ts';
import { Obstacle } from '../../Components/Obstacle.ts';

export type BuildingPartData = {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation: number;
    type: BuildingPartType;
};

// Ruined concrete colors - grays with dust
export const CONCRETE_COLORS: TColor[] = [
    new Float32Array([0.55, 0.53, 0.50, 1]),  // Light concrete
    new Float32Array([0.48, 0.46, 0.43, 1]),  // Medium concrete
    new Float32Array([0.40, 0.38, 0.36, 1]),  // Dark concrete
    new Float32Array([0.52, 0.50, 0.46, 1]),  // Dusty concrete
    new Float32Array([0.45, 0.43, 0.40, 1]),  // Weathered concrete
];

// Ruined brick colors - faded reds and browns
export const BRICK_COLORS: TColor[] = [
    new Float32Array([0.55, 0.35, 0.28, 1]),  // Faded red brick
    new Float32Array([0.50, 0.32, 0.25, 1]),  // Dark red brick
    new Float32Array([0.58, 0.40, 0.32, 1]),  // Light red brick
    new Float32Array([0.45, 0.30, 0.22, 1]),  // Old brick
    new Float32Array([0.52, 0.38, 0.30, 1]),  // Weathered brick
];

// Floor colors - concrete slabs and tiles
export const FLOOR_COLORS: TColor[] = [
    new Float32Array([0.42, 0.40, 0.38, 1]),  // Dark concrete floor
    new Float32Array([0.48, 0.45, 0.42, 1]),  // Medium concrete floor
    new Float32Array([0.38, 0.36, 0.34, 1]),  // Stained floor
    new Float32Array([0.44, 0.42, 0.40, 1]),  // Dusty floor
    new Float32Array([0.40, 0.38, 0.35, 1]),  // Cracked floor
];

export type BuildingMaterial = 'concrete' | 'brick';

export function getRandomBuildingColor(material: BuildingMaterial): TColor {
    const colors = material === 'brick' ? BRICK_COLORS : CONCRETE_COLORS;
    return colors[randomRangeInt(0, colors.length - 1)];
}

export function getRandomFloorColor(): TColor {
    return FLOOR_COLORS[randomRangeInt(0, FLOOR_COLORS.length - 1)];
}

// Building part creation options
export type BuildingPartOptions = {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    rotation: number;
    color: TColor;
    density: number;
};

const defaultPartOptions: BuildingPartOptions = {
    x: 0,
    y: 0,
    z: ZIndex.Building,
    width: 5,
    height: 5,
    rotation: 0,
    color: new Float32Array([0.5, 0.48, 0.45, 1]),
    density: 800,
};

const partOptions = { ...defaultPartOptions };

/**
 * Create building parts as fixed bodies
 */
export function createBuildingParts(
    buildingEid: EntityId,
    buildingX: number,
    buildingY: number,
    partsData: BuildingPartData[],
    material: BuildingMaterial,
    options: {
        rotation: number;
        density: number;
    },
    { world } = GameDI,
): EntityId[] {
    const partEids: EntityId[] = [];
    const baseColor = getRandomBuildingColor(material);

    for (let i = 0; i < partsData.length; i++) {
        const { x: anchorX, y: anchorY, width, height, rotation: partRotation, type } = partsData[i];

        // Transform anchor from local to world space
        const cos = Math.cos(options.rotation);
        const sin = Math.sin(options.rotation);
        const worldX = anchorX * cos - anchorY * sin;
        const worldY = anchorX * sin + anchorY * cos;

        // Choose color based on part type
        let partColor: TColor;
        if (type === 'floor') {
            partColor = getRandomFloorColor();
        } else {
            partColor = baseColor;
        }

        // Slightly vary the color for each part
        const colorVariation = randomRangeFloat(-0.04, 0.04);
        partOptions.color[0] = Math.max(0, Math.min(1, partColor[0] + colorVariation));
        partOptions.color[1] = Math.max(0, Math.min(1, partColor[1] + colorVariation));
        partOptions.color[2] = Math.max(0, Math.min(1, partColor[2] + colorVariation));
        // Floor tiles have random transparency for worn/faded look
        partOptions.color[3] = type === 'floor' ? randomRangeFloat(0.3, 0.8) : 1;

        partOptions.x = buildingX + worldX;
        partOptions.y = buildingY + worldY;
        
        // Z-ordering: floor (0, flat on ground) -> debris -> walls (highest)
        if (type === 'floor') {
            partOptions.z = 0;
        } else {
            partOptions.z = ZIndex.Building + 1 + (width + height) / 4;
        }

        partOptions.width = width;
        partOptions.height = height;
        // Combine building rotation with part's own rotation
        partOptions.rotation = options.rotation + partRotation;
        
        // Density: floor is heavy (flat on ground), debris is light, walls are normal
        if (type === 'floor') {
            partOptions.density = options.density * 1.2;
        } else {
            partOptions.density = options.density;
        }

        // Floor tiles have no collision - purely decorative
        const isFloor = type === 'floor';
        
        const rrOptions = {
            ...partOptions,
            bodyType: RigidBodyType.Fixed,
            belongsCollisionGroup: isFloor ? CollisionGroup.NONE : CollisionGroup.OBSTACLE,
            interactsCollisionGroup: isFloor ? CollisionGroup.NONE : CollisionGroup.ALL,
            belongsSolverGroup: isFloor ? CollisionGroup.NONE : CollisionGroup.ALL,
            interactsSolverGroup: isFloor ? CollisionGroup.NONE : CollisionGroup.ALL,
        };

        const [partEid, _partPid] = createRectangleRR(rrOptions);

        // Add components
        Obstacle.addComponent(world, partEid);
        BuildingPart.addComponent(world, partEid);
        Parent.addComponent(world, partEid, buildingEid);
        Children.addChildren(buildingEid, partEid);
        Color.addComponent(
            world,
            partEid,
            partOptions.color[0],
            partOptions.color[1],
            partOptions.color[2],
            partOptions.color[3],
        );

        // Floor tiles are not damageable - only walls and debris
        if (!isFloor) {
            // Make parts hitable and damageable
            // Walls are tougher than debris
            const healthMultiplier = type === 'wall' ? 1.5 : 0.5;
            const health = min(width, height) * 2 * healthMultiplier;
            Hitable.addComponent(world, partEid, health);
            Damagable.addComponent(world, partEid, min(width, height) / 8);
        }

        partEids.push(partEid);
    }

    return partEids;
}

