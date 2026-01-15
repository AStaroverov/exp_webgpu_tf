import * as tf from '@tensorflow/tfjs';
import { MAX_TURRETS } from "../../../tanks/src/Pilots/Components/TankState";
import { TANK_FEATURES_DIM, TURRET_FEATURES_DIM, RAY_SLOTS, RAY_FEATURES_DIM, ALLY_FEATURES_DIM, ALLY_SLOTS, BULLET_FEATURES_DIM, BULLET_SLOTS, ENEMY_FEATURES_DIM, ENEMY_SLOTS } from "./Create";
import { VEHICLE_TYPE_COUNT } from '../../../tanks/src/Game/Config';
import { createDenseLayer } from './ApplyLayers';

export function createInputs(name: string) {
    const tankInput = tf.input({name: name + '_tankInput', shape: [TANK_FEATURES_DIM]});
    const tankTypeInput = tf.input({name: name + '_tankTypeInput', shape: [1]});
    const turretInput = tf.input({name: name + '_turretInput', shape: [MAX_TURRETS, TURRET_FEATURES_DIM]});
    const raysInput = tf.input({name: name + '_raysInput', shape: [RAY_SLOTS, RAY_FEATURES_DIM]});
    
    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesTypesInput = tf.input({name: name + '_enemiesTypesInput', shape: [ENEMY_SLOTS]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS]});
    
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesTypesInput = tf.input({name: name + '_alliesTypesInput', shape: [ALLY_SLOTS]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [ALLY_SLOTS]});
    
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS]});

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
            // useBias: false, // bias is required for token type embedding
            useBias: true, // bias is required for token type embedding
            activation: 'linear',
        }).apply(input) as tf.SymbolicTensor;
        return typeEmbs.length > 0 ? tf.layers.add({ name: `${name}_concat` }).apply([token, ...typeEmbs]) as tf.SymbolicTensor : token;
    };

    const to3D = (x: tf.SymbolicTensor): tf.SymbolicTensor => {
        const lastDim = x.shape[x.shape.length - 1] as number;
        return tf.layers.reshape({
            name: `${x.name}_3d`,
            targetShape: [1, lastDim],
        }).apply(x) as tf.SymbolicTensor;
    };

    const vehicleTypeEmbedding = tf.layers.embedding({
        name: 'vehicleTypeEmbedding',
        inputDim: VEHICLE_TYPE_COUNT,
        outputDim: dModel,
        embeddingsInitializer: 'zeros',
    });

    // Tank
    const tankInput3D = to3D(tankInput);
    const tankVehicleEmb = vehicleTypeEmbedding.apply(tankTypeInput) as tf.SymbolicTensor;
    const tankTok = toToken('tank', tankInput3D, tankVehicleEmb); //tankVehicleEmb

    // Turret
    const turretTok = toToken('turret', turretInput); //tankVehicleEmb

    // Rays
    const raysTok = toToken('rays', raysInput);

    // Enemies
    const enemiesVehicleEmb = vehicleTypeEmbedding.apply(enemiesTypesInput) as tf.SymbolicTensor;
    const enemiesTok = toToken('enemies', enemiesInput, enemiesVehicleEmb);

    // Allies
    const alliesVehicleEmb = vehicleTypeEmbedding.apply(alliesTypesInput) as tf.SymbolicTensor;
    const alliesTok = toToken('allies', alliesInput, alliesVehicleEmb);

    // Bullets
    const bulletsTok = toToken('bullets', bulletsInput);
    
    return {
        tankTok,
        turretTok,
        raysTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
    };
}
