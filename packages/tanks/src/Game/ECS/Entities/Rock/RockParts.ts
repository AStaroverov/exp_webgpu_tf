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
import { RockPart } from '../../Components/Rock.ts';
import { Obstacle } from '../../Components/Obstacle.ts';

export type RockPartData = [x: number, y: number, w: number, h: number, rotation: number];

// Stone colors - various shades of gray and brown
export const STONE_COLORS: TColor[] = [
    new Float32Array([0.35, 0.33, 0.30, 1]),  // Dark gray-brown
    new Float32Array([0.42, 0.40, 0.36, 1]),  // Medium gray
    new Float32Array([0.38, 0.35, 0.32, 1]),  // Gray-brown
    new Float32Array([0.45, 0.42, 0.38, 1]),  // Light gray
    new Float32Array([0.32, 0.30, 0.28, 1]),  // Dark stone
    new Float32Array([0.40, 0.38, 0.35, 1]),  // Medium stone
    new Float32Array([0.48, 0.45, 0.40, 1]),  // Lighter stone
    new Float32Array([0.36, 0.34, 0.30, 1]),  // Brownish gray
];

export function getRandomStoneColor(): TColor {
    return STONE_COLORS[randomRangeInt(0, STONE_COLORS.length - 1)];
}

// Rock creation options for parts
export type RockPartOptions = {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    rotation: number;
    color: TColor;
    density: number;
};

const defaultPartOptions: RockPartOptions = {
    x: 0,
    y: 0,
    z: ZIndex.Rock,
    width: 5,
    height: 5,
    rotation: 0,
    color: new Float32Array([0.4, 0.38, 0.35, 1]),
    density: 500,
};

const partOptions = { ...defaultPartOptions };

/**
 * Create rock parts as fixed bodies (no joints needed)
 */
export function createRockParts(
    rockEid: EntityId,
    rockX: number,
    rockY: number,
    partsData: RockPartData[],
    baseColor: TColor,
    options: {
        rotation: number;
        density: number;
    },
    { world } = GameDI,
): EntityId[] {
    const partEids: EntityId[] = [];

    for (let i = 0; i < partsData.length; i++) {
        const [anchorX, anchorY, width, height, partRotation] = partsData[i];

        // Transform anchor from local to world space
        const cos = Math.cos(options.rotation);
        const sin = Math.sin(options.rotation);
        const worldX = anchorX * cos - anchorY * sin;
        const worldY = anchorX * sin + anchorY * cos;

        // Slightly vary the color for each part
        const colorVariation = randomRangeFloat(-0.05, 0.05);
        partOptions.color[0] = Math.max(0, Math.min(1, baseColor[0] + colorVariation));
        partOptions.color[1] = Math.max(0, Math.min(1, baseColor[1] + colorVariation));
        partOptions.color[2] = Math.max(0, Math.min(1, baseColor[2] + colorVariation));
        partOptions.color[3] = 1;

        partOptions.x = rockX + worldX;
        partOptions.y = rockY + worldY;
        partOptions.z = ZIndex.Rock + (width + height) / 2;
        
        partOptions.width = width;
        partOptions.height = height;
        // Combine rock rotation with part's own rotation
        partOptions.rotation = options.rotation + partRotation;
        partOptions.density = options.density;

        const rrOptions = {
            ...partOptions,
            bodyType: RigidBodyType.Fixed,
            belongsCollisionGroup: CollisionGroup.OBSTACLE,
            interactsCollisionGroup: CollisionGroup.ALL,
            belongsSolverGroup: CollisionGroup.ALL,
            interactsSolverGroup: CollisionGroup.ALL,
        };

        const [partEid, _partPid] = createRectangleRR(rrOptions);
       
        // Add components
        Obstacle.addComponent(world, partEid);
        RockPart.addComponent(world, partEid);
        Parent.addComponent(world, partEid, rockEid);
        Children.addChildren(rockEid, partEid);
        Color.addComponent(world, partEid, partOptions.color[0], partOptions.color[1], partOptions.color[2], partOptions.color[3]);

        // Make parts hitable and damageable
        const health = min(width, height) * 2;
        Hitable.addComponent(world, partEid, health);
        Damagable.addComponent(world, partEid, min(width, height) / 10);

        partEids.push(partEid);
    }

    return partEids;
}
