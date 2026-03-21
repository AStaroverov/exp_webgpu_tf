import * as tf from '@tensorflow/tfjs';
import { MAX_TURRETS } from "../../../tanks/src/Plugins/Pilots/Components/TankState";
import { TANK_FEATURES_DIM, TURRET_FEATURES_DIM, RAY_SLOTS, RAY_FEATURES_DIM, ALLY_FEATURES_DIM, ALLY_SLOTS, BULLET_FEATURES_DIM, BULLET_SLOTS, ENEMY_FEATURES_DIM, ENEMY_SLOTS } from "./Create";
import { VEHICLE_TYPE_COUNT } from '../../../tanks/src/Game/Config';
import { createDenseLayer } from './ApplyLayers';
import { HISTORY_LENGTH } from '../../../ml-common/historyConfig';
import { TemporalPositionLayer } from './Layers/TemporalPositionLayer';

const T = HISTORY_LENGTH;

export function createInputs(name: string) {
    // Tank features: [B, T, TANK_FEATURES_DIM] — T tokens (one per temporal frame)
    const tankInput = tf.input({name: name + '_tankInput', shape: [T, TANK_FEATURES_DIM]});
    // Tank type: [B, T] — one type per frame
    const tankTypeInput = tf.input({name: name + '_tankTypeInput', shape: [T]});
    // Turret: [B, T * MAX_TURRETS, TURRET_FEATURES_DIM]
    const turretInput = tf.input({name: name + '_turretInput', shape: [T * MAX_TURRETS, TURRET_FEATURES_DIM]});
    // Rays: [B, T * RAY_SLOTS, RAY_FEATURES_DIM]
    const raysInput = tf.input({name: name + '_raysInput', shape: [T * RAY_SLOTS, RAY_FEATURES_DIM]});

    // Enemies: [B, T * ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [T * ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesTypesInput = tf.input({name: name + '_enemiesTypesInput', shape: [T * ENEMY_SLOTS]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [T * ENEMY_SLOTS]});

    // Allies: [B, T * ALLY_SLOTS, ALLY_FEATURES_DIM]
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [T * ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesTypesInput = tf.input({name: name + '_alliesTypesInput', shape: [T * ALLY_SLOTS]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [T * ALLY_SLOTS]});

    // Bullets: [B, T * BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [T * BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [T * BULLET_SLOTS]});

    return {
        tankInput,
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
    };
}

export function convertInputsToTokens(
    {
        tankInput,
        tankTypeInput,
        turretInput,
        raysInput,
        enemiesInput,
        enemiesTypesInput,
        alliesInput,
        alliesTypesInput,
        bulletsInput,
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

    const addTemporalEncoding = (name: string, token: tf.SymbolicTensor, slotsPerFrame: number): tf.SymbolicTensor => {
        return new TemporalPositionLayer({
            name: `${name}_temporalPos`,
            dModel,
            slotsPerFrame,
        }).apply(token) as tf.SymbolicTensor;
    };

    const vehicleTypeEmbedding = tf.layers.embedding({
        name: 'vehicleTypeEmbedding',
        inputDim: VEHICLE_TYPE_COUNT,
        outputDim: dModel,
        embeddingsInitializer: 'zeros',
    });

    // Tank: already [B, T, TANK_FEATURES_DIM] — 3D, no need for to3D
    const tankVehicleEmb = vehicleTypeEmbedding.apply(tankTypeInput) as tf.SymbolicTensor; // [B, T, dModel]
    const tankTok = addTemporalEncoding('tank', toToken('tank', tankInput, tankVehicleEmb), 1); // [B, T, dModel]

    // Turret: [B, T * MAX_TURRETS, TURRET_FEATURES_DIM]
    const turretTok = addTemporalEncoding('turret', toToken('turret', turretInput), MAX_TURRETS);

    // Rays: [B, T * RAY_SLOTS, RAY_FEATURES_DIM]
    const raysTok = addTemporalEncoding('rays', toToken('rays', raysInput), RAY_SLOTS);

    // Enemies: [B, T * ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesVehicleEmb = vehicleTypeEmbedding.apply(enemiesTypesInput) as tf.SymbolicTensor;
    const enemiesTok = addTemporalEncoding('enemies', toToken('enemies', enemiesInput, enemiesVehicleEmb), ENEMY_SLOTS);

    // Allies: [B, T * ALLY_SLOTS, ALLY_FEATURES_DIM]
    const alliesVehicleEmb = vehicleTypeEmbedding.apply(alliesTypesInput) as tf.SymbolicTensor;
    const alliesTok = addTemporalEncoding('allies', toToken('allies', alliesInput, alliesVehicleEmb), ALLY_SLOTS);

    // Bullets: [B, T * BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsTok = addTemporalEncoding('bullets', toToken('bullets', bulletsInput), BULLET_SLOTS);

    return {
        tankTok,
        turretTok,
        raysTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
    };
}
