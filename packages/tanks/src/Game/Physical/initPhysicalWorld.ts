import { Vector2, World } from '@dimforge/rapier2d-simd';
import { createRigidRectangle } from './createRigid';
import { RawIntegrationParameters } from '@dimforge/rapier2d-simd/rapier_wasm2d';

export function initPhysicalWorld() {
    const gravity = new Vector2(0, 0);
    const integrationParameters = new RawIntegrationParameters();
    integrationParameters.lengthUnit = 100;
    integrationParameters.numSolverIterations = 4;
    const world = new World(gravity, integrationParameters);

    // skip id == 0 because it's the default value for empty memory data
    createRigidRectangle(
        { enabled: false, width: 10, height: 10, x: 0, y: 0 },
        // @ts-ignore
        { physicalWorld: world }
    );

    return world;
}

export type PhysicalWorld = World;