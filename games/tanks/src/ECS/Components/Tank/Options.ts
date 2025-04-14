import { RigidBodyType } from '@dimforge/rapier2d-simd';
import { TColor } from '../../../../../../src/ECS/Components/Common.ts';
import { createTank } from './CreateTank.ts';

export type Options = typeof mutatedOptions;

export const BASE_DENSITY = 300;

export const mutatedOptions = {
    x: 0,
    y: 0,
    z: 0,
    width: 0,
    height: 0,
    radius: 0,
    rotation: 0,
    color: new Float32Array([1, 0, 0, 1]),
    shadow: new Float32Array([0, 3]),
    bodyType: RigidBodyType.Dynamic,
    density: 0,
    linearDamping: 5,
    angularDamping: 5,
    belongsSolverGroup: 0,
    interactsSolverGroup: 0,
    belongsCollisionGroup: 0,
    interactsCollisionGroup: 0,

    teamId: 0,
    playerId: 0,
};

export const defaultOptions = structuredClone(mutatedOptions);
export const resetOptions = (target: Options, source: Parameters<typeof createTank>[0]) => {
    target.x = source?.x ?? defaultOptions.x;
    target.y = source?.y ?? defaultOptions.y;
    target.z = defaultOptions.z;
    target.width = defaultOptions.width;
    target.height = defaultOptions.height;
    target.radius = defaultOptions.radius;
    target.rotation = source?.rotation ?? defaultOptions.rotation;
    (target.color as Float32Array).set(source?.color ?? defaultOptions.color, 0);
    (target.shadow as Float32Array).set(defaultOptions.shadow, 0);
    target.density = defaultOptions.density;
    target.linearDamping = defaultOptions.linearDamping;
    target.angularDamping = defaultOptions.angularDamping;
    target.belongsSolverGroup = defaultOptions.belongsSolverGroup;
    target.interactsSolverGroup = defaultOptions.interactsSolverGroup;
    target.belongsCollisionGroup = defaultOptions.belongsCollisionGroup;
    target.interactsCollisionGroup = defaultOptions.interactsCollisionGroup;

    target.teamId = source?.teamId ?? defaultOptions.teamId;
    target.playerId = source?.playerId ?? defaultOptions.playerId;
};
export const updateColorOptions = (target: Options, color: TColor) => {
    (target.color as Float32Array).set(color, 0);
};