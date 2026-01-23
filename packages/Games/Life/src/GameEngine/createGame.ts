import { createWorld, deleteWorld, EntityId, resetWorld } from 'bitecs';
import { GameDI } from './DI/GameDI.js';
import { PlayerEnvDI } from './DI/PlayerEnvDI.js';
import { RenderDI } from './DI/RenderDI.js';
import { SystemGroup } from './ECS/Plugins/systems.js';
import { createDestroyByTimeoutSystem } from './ECS/Systems/createDestroyByTimeoutSystem.js';
import { createDestroySystem } from './ECS/Systems/createDestroySystem.js';
import { createProgressSystem } from './ECS/Systems/createProgressSystem.js';
import { createHitableSystem } from './ECS/Systems/createHitableSystem.js';
import { SoundDI } from './DI/SoundDI.js';
import { destroyChangeDetectorSystem } from 'renderer/src/ECS/Systems/ChangedDetectorSystem.js';
import { RigidBodyRef } from '../../../../tanks/src/Game/ECS/Components/Physical.js';
import { initWebGPU } from 'renderer/src/gpu.js';
import { createFrameTextures } from 'renderer/src/WGSL/createFrame.js';
import { createDrawShapeSystem } from 'renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.js';
import { createFrameTick } from 'renderer/src/WGSL/createFrame.js';
import { setCameraPosition } from 'renderer/src/ECS/Systems/ResizeSystem.js';
import { createLifeSystem } from './ECS/Systems/LifeSystem.js';
import { CELL_SIZE } from './ECS/Entities/Cell.js';
import { createSpawnerSystem } from './ECS/Systems/SpawnerSystem.js';
import { createMouseInputSystem } from './ECS/Systems/MouseInputSystem.js';
import { createTransformSystem } from 'renderer/src/ECS/Systems/TransformSystem.js';
import { createPostEffect } from '../../../../tanks/src/Game/ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.js';

export type Game = ReturnType<typeof createGame>;

export function createGame({ cells, rows }: {
    cells: number,
    rows: number,
}) {
    const world = createWorld();

    GameDI.cells = cells;
    GameDI.rows = rows;
    GameDI.world = world;

    const destroy = createDestroySystem();
    const destroyByTimeout = createDestroyByTimeoutSystem();

    const destroyFrame = (delta: number) => {
        destroyByTimeout(delta);
        destroy();
    };

    const execTransformSystem = createTransformSystem(world);
    const updateProgress = createProgressSystem();
    const updateHitableSystem = createHitableSystem();
    
    // Game of Life systems
    const lifeSystem = createLifeSystem();
    const spawnerSystem = createSpawnerSystem();
    const mouseInputSystem = createMouseInputSystem();

    GameDI.gameTick = (delta: number) => {
        if (GameDI.world === null) return;

        GameDI.plugins.systems[SystemGroup.Before].forEach(system => system(delta));

        // Mouse input for creating spawners
        mouseInputSystem.tick(delta);

        execTransformSystem();
        
        // Spawner system - creates new cells
        spawnerSystem(delta);
        
        // Life system - Game of Life rules
        lifeSystem(delta);

        updateHitableSystem(delta);
        updateProgress(delta);

        destroyFrame(delta);

        PlayerEnvDI.inputFrame?.();
        RenderDI.renderFrame?.(delta);
        SoundDI.soundFrame?.(delta);

        GameDI.plugins.systems[SystemGroup.After].forEach(system => system(delta));
    };

    GameDI.destroy = () => {
        GameDI.plugins.dispose();

        RigidBodyRef.dispose();
        
        // Dispose mouse input system
        mouseInputSystem.dispose();

        resetWorld(world);
        deleteWorld(world);
        destroyChangeDetectorSystem(world);

        GameDI.cells = null!;
        GameDI.rows = null!;
        GameDI.world = null!;
        GameDI.gameTick = null!;
        GameDI.destroy = null!;

        SoundDI.destroy?.();
        RenderDI.destroy?.();
        PlayerEnvDI.destroy?.();
    };

    // GameDI.enableSound = async () => {
    //     if (SoundDI.enabled) {
    //         return;
    //     }

    //     SoundDI.enabled = true;

    //     const updateSounds = createSoundSystem();
    //     const updateTankMoveSounds = createTankMoveSoundSystem();

    //     // Load sounds asynchronously
    //     loadGameSounds().catch(console.error);

    //     SoundDI.soundFrame = (delta: number) => {
    //         updateSounds(delta);
    //         updateTankMoveSounds(delta);
    //     };

    //     SoundDI.destroy = () => {
    //         disposeSoundSystem();
    //         SoundManager.dispose();

    //         SoundDI.enabled = false;
    //         SoundDI.destroy = undefined;
    //         SoundDI.soundFrame = undefined;
    //     };
    // }

    GameDI.setRenderTarget = async (canvas: null | undefined | HTMLCanvasElement) => {
        if (canvas === RenderDI.canvas) {
            return;
        }

        RenderDI.destroy?.();
        RenderDI.enabled = canvas != null;

        if (canvas == null) {
            return;
        }

        RenderDI.canvas = canvas;

        // Center camera on the grid
        const gridWorldSize = cells * CELL_SIZE;
        setCameraPosition(gridWorldSize / 2, gridWorldSize / 2);

        const { device, context } = await initWebGPU(canvas);
        RenderDI.device = device;
        RenderDI.context = context;

        const textures = createFrameTextures(device, canvas);

        const shapeSystem = createDrawShapeSystem({
            world,
            device,
            shadowMapTexture: textures.shadowMapTexture,
        });
        
        // First create frame tick to get shadowMapTexture
        const frameTick = createFrameTick(
            {
                ...textures,
                canvas,
                device,
                background: [0.05, 0.05, 0.1, 1], // Dark blue-black background
                getPixelRatio: () => window.devicePixelRatio,
            }, 
            // Main render pass callback
            ({ passEncoder }) => {
                shapeSystem.drawShapes(passEncoder);
            },
            // Shadow map pass callback
            ({ passEncoder: shadowMapPassEncoder }) => {
                shapeSystem.drawShadowMap(shadowMapPassEncoder);
            }
        );
        
        // Create shape system with shadowMapTexture
        const postEffectFrame = createPostEffect(device, context, textures.renderTexture);

        RenderDI.renderFrame = (delta: number) => {
            const commandEncoder = device.createCommandEncoder();
            frameTick(commandEncoder, delta);
            postEffectFrame(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        };

        RenderDI.destroy = () => {
            RenderDI.enabled = false;
            RenderDI.canvas = null!;
            RenderDI.device = null!;
            RenderDI.context = null!;
            RenderDI.renderFrame = null!;
        };
    };

    GameDI.enablePlayer = () => {
        PlayerEnvDI.destroy?.();

        PlayerEnvDI.inputFrame = () => {
        };

        PlayerEnvDI.destroy = () => {
            PlayerEnvDI.playerId = null;
            PlayerEnvDI.destroy = null!;
            PlayerEnvDI.inputFrame = null!;
        };
    };
    GameDI.setPlayerId = (playerId: null | EntityId) => {
        PlayerEnvDI.playerId = playerId;
    };

    return GameDI;
}