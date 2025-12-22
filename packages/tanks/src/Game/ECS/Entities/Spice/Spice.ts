import { ActiveCollisionTypes, RigidBodyType } from '@dimforge/rapier2d-simd';
import { EntityId } from 'bitecs';
import { randomRangeFloat, randomRangeInt } from '../../../../../../../lib/random.ts';
import { Color, TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { SpiceConfig, SpiceColors, SpicePhysicsConfig } from '../../../Config/index.ts';
import { createRectangleRR } from '../../Components/RigidRender.ts';
import { Spice } from '../../Components/Spice.ts';

export function getRandomSpiceColor(): TColor {
    return SpiceColors[randomRangeInt(0, SpiceColors.length - 1)];
}

/**
 * Options for creating a single spice piece
 */
export type CreateSpiceOptions = {
    x: number;
    y: number;
    size?: number;
    color?: TColor;
    value?: number;
};

/**
 * Create a single spice piece entity.
 */
export function createSpice(opts: CreateSpiceOptions, { world } = GameDI): EntityId {
    const size = opts.size ?? randomRangeInt(...SpiceConfig.size);
    const color = opts.color ?? getRandomSpiceColor();

    // Slightly vary the color for visual variety
    const colorVariation = randomRangeFloat(-0.08, 0.08);
    const variedColor = new Float32Array([
        Math.max(0, Math.min(1, color[0] + colorVariation)),
        Math.max(0, Math.min(1, color[1] + colorVariation)),
        Math.max(0, Math.min(1, color[2] + colorVariation)),
        1,
    ]);

    const rrOptions = {
        x: opts.x,
        y: opts.y,
        z: SpicePhysicsConfig.z,
        width: size,
        height: size,
        color: variedColor,
        rotation: randomRangeFloat(0, Math.PI * 2),
        bodyType: RigidBodyType.Dynamic,
        density: size * size * SpiceConfig.density,
        linearDamping: SpiceConfig.damping,
        angularDamping: SpiceConfig.damping,
        belongsCollisionGroup: SpicePhysicsConfig.belongsCollisionGroup,
        interactsCollisionGroup: SpicePhysicsConfig.interactsCollisionGroup,
        belongsSolverGroup: SpicePhysicsConfig.belongsSolverGroup,
        interactsSolverGroup: SpicePhysicsConfig.interactsSolverGroup,
        // Enable collision detection with fixed sensors (spice collectors)
        activeCollisionTypes: ActiveCollisionTypes.DEFAULT | ActiveCollisionTypes.DYNAMIC_FIXED,
    };

    const [spiceEid] = createRectangleRR(rrOptions);
    
    // Add components
    Spice.addComponent(world, spiceEid);
    Color.addComponent(world, spiceEid, variedColor[0], variedColor[1], variedColor[2], variedColor[3]);

    return spiceEid;
}

/**
 * Options for spawning a cluster of spice
 */
export type SpawnSpiceClusterOptions = {
    /** Center X position of the cluster */
    x: number;
    /** Center Y position of the cluster */
    y: number;
    /** Number of spice pieces to spawn (default: 10-50) */
    count?: number;
    /** Spread radius around center (default: 20-40) */
    spread?: number;
    /** Base color for all pieces in cluster (default: random) */
    color?: TColor;
};

/**
 * Spawn a cluster of spice pieces around a center point.
 * Spice pieces are spawned with gaussian-like distribution (more dense in center).
 */
export function spawnSpiceCluster(opts: SpawnSpiceClusterOptions): EntityId[] {
    const count = opts.count ?? randomRangeInt(...SpiceConfig.countRange);
    const spread = opts.spread ?? randomRangeFloat(...SpiceConfig.spreadRange);
    const baseColor = opts.color ?? getRandomSpiceColor();
    
    const spiceEids: EntityId[] = [];

    for (let i = 0; i < count; i++) {
        // Use gaussian-like distribution for more natural clustering
        // Box-Muller transform approximation with multiple uniform randoms
        const r1 = randomRangeFloat(0, 1);
        const r2 = randomRangeFloat(0, 1);
        const r3 = randomRangeFloat(0, 1);
        const gaussianFactor = (r1 + r2 + r3) / 3; // Tends toward 0.5

        const angle = randomRangeFloat(0, Math.PI * 2);
        const distance = gaussianFactor * spread;
        
        const x = opts.x + Math.cos(angle) * distance;
        const y = opts.y + Math.sin(angle) * distance;

        const spiceEid = createSpice({
            x,
            y,
            color: baseColor,
        });

        spiceEids.push(spiceEid);
    }

    return spiceEids;
}

