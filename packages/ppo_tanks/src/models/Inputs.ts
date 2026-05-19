import * as tf from '@tensorflow/tfjs';
import { MAX_TURRETS } from "../../../tanks/src/Plugins/Pilots/Components/TankState";
import { TANK_FEATURES_DIM, TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM, TURRET_FEATURES_DIM, RAY_SLOTS, RAY_FEATURES_DIM, ALLY_FEATURES_DIM, ALLY_SLOTS, BULLET_FEATURES_DIM, BULLET_SLOTS, ENEMY_FEATURES_DIM, ENEMY_SLOTS, GRID_CELLS, GRID_CELL_FEATURES, GRID_SIZE } from "./dims.ts";
import { VEHICLE_TYPE_COUNT } from '../../../tanks/src/Game/Config';
import { createDenseLayer } from '../../../ppo/src/models/ApplyLayers.ts';

export function createInputs(name: string) {
    // Tank features: [B, 1, TANK_FEATURES_DIM]
    const tankInput = tf.input({name: name + '_tankInput', shape: [1, TANK_FEATURES_DIM]});
    // Tank history: [B, TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM]
    const tankHistoryInput = tf.input({name: name + '_tankHistoryInput', shape: [TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM]});
    // Tank type: [B, 1]
    const tankTypeInput = tf.input({name: name + '_tankTypeInput', shape: [1]});
    // Turret: [B, MAX_TURRETS, TURRET_FEATURES_DIM]
    const turretInput = tf.input({name: name + '_turretInput', shape: [MAX_TURRETS, TURRET_FEATURES_DIM]});
    // Rays: [B, RAY_SLOTS, RAY_FEATURES_DIM]
    const raysInput = tf.input({name: name + '_raysInput', shape: [RAY_SLOTS, RAY_FEATURES_DIM]});

    // Enemies: [B, ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesTypesInput = tf.input({name: name + '_enemiesTypesInput', shape: [ENEMY_SLOTS]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS]});

    // Allies: [B, ALLY_SLOTS, ALLY_FEATURES_DIM]
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesTypesInput = tf.input({name: name + '_alliesTypesInput', shape: [ALLY_SLOTS]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [ALLY_SLOTS]});

    // Bullets: [B, BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS]});

    // Obstacle grid: [B, GRID_CELLS, GRID_CELL_FEATURES] — static per episode
    const obstacleGridInput = tf.input({name: name + '_obstacleGridInput', shape: [GRID_CELLS, GRID_CELL_FEATURES]});

    return {
        tankInput,
        tankHistoryInput,
        tankTypeInput,
        turretInput,
        raysInput,

        enemiesInput,
        enemiesTypesInput,
        enemiesMaskInput,

        alliesInput,
        alliesTypesInput,
        alliesMaskInput,

        bulletsInput,
        bulletsMaskInput,

        obstacleGridInput,
    };
}

export function convertInputsToTokens(
    {
        tankInput,
        tankHistoryInput,
        tankTypeInput,
        turretInput,
        enemiesInput,
        enemiesTypesInput,
        alliesInput,
        alliesTypesInput,
        bulletsInput,
        raysInput,
        obstacleGridInput,
    }: ReturnType<typeof createInputs>,
    dModel: number,
) {
    const toToken = (
        name: string,
        input: tf.SymbolicTensor,
        ...typeEmbs: tf.SymbolicTensor[]
    ): tf.SymbolicTensor => {
        const token =  createDenseLayer({
            name: `${name}_tokEmbedding`,
            units: dModel,
            useBias: true,
            activation: 'linear',
        }).apply(input) as tf.SymbolicTensor;
        return typeEmbs.length > 0 ? tf.layers.add({ name: `${name}_concat` }).apply([token, ...typeEmbs]) as tf.SymbolicTensor : token;
    };

    const vehicleTypeEmbedding = tf.layers.embedding({
        name: 'vehicleTypeEmbedding',
        inputDim: VEHICLE_TYPE_COUNT,
        outputDim: dModel,
        embeddingsInitializer: 'zeros',
    });

    // Tank: [B, 1, TANK_FEATURES_DIM]
    const tankVehicleEmb = vehicleTypeEmbedding.apply(tankTypeInput) as tf.SymbolicTensor;
    const tankTok = toToken('tank', tankInput, tankVehicleEmb);

    // Tank history: [B, TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM]
    const tankHistoryTok = toToken('tankHistory', tankHistoryInput);

    // Turret: [B, MAX_TURRETS, TURRET_FEATURES_DIM]
    const turretTok = toToken('turret', turretInput);

    // Enemies: [B, ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesVehicleEmb = vehicleTypeEmbedding.apply(enemiesTypesInput) as tf.SymbolicTensor;
    const enemiesTok = toToken('enemies', enemiesInput, enemiesVehicleEmb);

    // Allies: [B, ALLY_SLOTS, ALLY_FEATURES_DIM]
    const alliesVehicleEmb = vehicleTypeEmbedding.apply(alliesTypesInput) as tf.SymbolicTensor;
    const alliesTok = toToken('allies', alliesInput, alliesVehicleEmb);

    // Bullets: [B, BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsTok = toToken('bullets', bulletsInput);

    // Rays patching: [B, 128, 4] → reshape [B, 16, 32] → Dense → [B, 16, dModel]
    const RAY_PATCH_SIZE = 8;
    const raysPatched = tf.layers.reshape({
        name: 'rays_patch',
        targetShape: [RAY_SLOTS / RAY_PATCH_SIZE, RAY_FEATURES_DIM * RAY_PATCH_SIZE],
    }).apply(raysInput) as tf.SymbolicTensor;
    const raysTok = toToken('rays', raysPatched);

    // Grid block patching: [B, 256, 3] → blocks 4×4 → [B, 16, 48] → Dense → [B, 16, dModel]
    const GRID_PATCH_SIZE = 4;
    const gridSpatial = tf.layers.reshape({
        name: 'grid_toSpatial',
        targetShape: [GRID_SIZE, GRID_SIZE, GRID_CELL_FEATURES],
    }).apply(obstacleGridInput) as tf.SymbolicTensor;
    const gridBlocks = tf.layers.reshape({
        name: 'grid_toBlocks',
        targetShape: [GRID_SIZE / GRID_PATCH_SIZE, GRID_PATCH_SIZE, GRID_SIZE / GRID_PATCH_SIZE, GRID_PATCH_SIZE, GRID_CELL_FEATURES],
    }).apply(gridSpatial) as tf.SymbolicTensor;
    const gridPermuted = tf.layers.permute({
        name: 'grid_permuteBlocks',
        dims: [1, 3, 2, 4, 5],
    }).apply(gridBlocks) as tf.SymbolicTensor;
    const gridPatched = tf.layers.reshape({
        name: 'grid_patch',
        targetShape: [(GRID_SIZE / GRID_PATCH_SIZE) ** 2, GRID_PATCH_SIZE ** 2 * GRID_CELL_FEATURES],
    }).apply(gridPermuted) as tf.SymbolicTensor;
    const gridTok = toToken('grid', gridPatched);

    return {
        tankTok,
        tankHistoryTok,
        turretTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
        raysTok,
        gridTok,
    };
}
