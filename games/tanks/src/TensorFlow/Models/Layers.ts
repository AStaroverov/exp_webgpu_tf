import * as tf from '@tensorflow/tfjs';
import { AttentionMaskLayer } from './AttentionMaskLayer.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { OnesMask } from './ConstOnesMaskLayer.ts';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { MultiHeadSelfAttentionLayer } from './MultiHeadSelfAttentionLayer.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    TANK_FEATURES_DIM,
} from './Create.ts';
import { CrossAttentionLayer } from './CrossAttentionLayer.ts';

export function createInputs(name: string) {
    const battleInput = tf.input({ name: name + '_battlefieldInput', shape: [BATTLE_FEATURES_DIM] });
    const tankInput = tf.input({ name: name + '_tankInput', shape: [TANK_FEATURES_DIM] });

    const enemiesInput = tf.input({ name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const enemiesMaskInput = tf.input({ name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS] });
    const alliesInput = tf.input({ name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM] });
    const alliesMaskInput = tf.input({ name: name + '_alliesMaskInput', shape: [ALLY_SLOTS] });
    const bulletsInput = tf.input({ name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });
    const bulletsMaskInput = tf.input({ name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS] });

    return {
        battleInput,
        tankInput,
        enemiesInput,
        enemiesMaskInput,
        alliesInput,
        alliesMaskInput,
        bulletsInput,
        bulletsMaskInput,
    };
}

export function applyAttentionLayer(name: string, qInput: tf.SymbolicTensor, kvInput: tf.SymbolicTensor, kvMaskInput: tf.SymbolicTensor, dModel = 32): tf.SymbolicTensor {
    const denseQ = tf.layers.dense({ name: name + '_denseQ', units: dModel, useBias: false });
    const denseK = tf.layers.dense({ name: name + '_denseK', units: dModel, useBias: false });
    const denseV = tf.layers.dense({ name: name + '_denseV', units: dModel, useBias: false });

    const Q = denseQ.apply(qInput) as tf.SymbolicTensor;  // [batch, dim model]
    const K = denseK.apply(kvInput) as tf.SymbolicTensor;  // [batch, slots, dim model]
    const V = denseV.apply(kvInput) as tf.SymbolicTensor;

    const Q_reshaped = tf.layers.reshape({ targetShape: [1, dModel] }).apply(Q) as tf.SymbolicTensor; // [batch, 1, d_model]

    const scores = tf.layers.dot({ axes: -1 }).apply([Q_reshaped, K]) as tf.SymbolicTensor;

    const maskedScores = new AttentionMaskLayer({ name: name + '_AttentionMaskLayer' })
        .apply([scores, kvMaskInput]) as tf.SymbolicTensor;

    const attnWeights = tf.layers.activation({ activation: 'softmax' }).apply(maskedScores) as tf.SymbolicTensor;

    const context = tf.layers.dot({ axes: [2, 1] }).apply([attnWeights, V]) as tf.SymbolicTensor;

    return tf.layers.flatten().apply(context) as tf.SymbolicTensor; // [batch, d_model]
}

export function applyDenseLayers(layer: tf.SymbolicTensor, hiddenLayers: [ActivationIdentifier, number][]) {
    let x = layer;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `${ layer.name }/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}

function proj(x: tf.SymbolicTensor, dModel: number, name: string) {
    return tf.layers.dense({ units: dModel, useBias: true, name: name + '_tokProj' }).apply(x);
}

export function applySelfAttentionLayer(
    name: string,
    dModel: number, // 32 | 64 |....
    numHeads: number,
    {
        battleInput,
        tankInput,
        alliesInput,
        enemiesInput,
        bulletsInput,
        alliesMaskInput,
        enemiesMaskInput,
        bulletsMaskInput,
    }: ReturnType<typeof createInputs>,
) {
    name = name + '_applySelfAttentionLayer';

    // ---------- embed everything to a common d_model ---------------------------
    const battleTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(battleInput, dModel, name + '_' + battleInput.name));
    const tankTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(tankInput, dModel, name + '_' + tankInput.name));
    const allyTok = proj(alliesInput, dModel, name + '_' + alliesInput.name);
    const enemyTok = proj(enemiesInput, dModel, name + '_' + enemiesInput.name);
    const bulletTok = proj(bulletsInput, dModel, name + '_' + bulletsInput.name);

    const tokens = tf.layers.concatenate({ name: name + '_tokens', axis: 1 })
        .apply([battleTok, tankTok, allyTok, enemyTok, bulletTok] as tf.SymbolicTensor[]) as tf.SymbolicTensor;            // [B,S,d]

    // ---------- build 0/1 padding mask -----------------------------------------
    const battleInputFixedMask = new OnesMask({ name: name + '_battleInputFixedMask' }).apply(battleInput) as tf.SymbolicTensor;   // [dModel,2]
    const tankInputFixedMask = new OnesMask({ name: name + '_tankInputFixedMask' }).apply(tankInput) as tf.SymbolicTensor;   // [dModel,2]
    const mask = tf.layers.concatenate({ name: name + '_mask', axis: 1 })
        .apply([
            battleInputFixedMask,
            tankInputFixedMask,
            alliesMaskInput,
            enemiesMaskInput,
            bulletsMaskInput,
        ] as SymbolicTensor[]) as tf.SymbolicTensor;           // [B,S]

    // ---------- self-attention block -------------------------------------------
    let x = new MultiHeadSelfAttentionLayer({
        name: name + '_MultiHeadSelfAttentionLayer',
        numHeads: numHeads,
        keyDim: dModel / numHeads,
    }).apply([tokens, mask]) as tf.SymbolicTensor;

    x = tf.layers.add({ name: name + '_add' }).apply([x, tokens]) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: name + '_normalization', epsilon: 1e-5 }).apply(x) as tf.SymbolicTensor;

    return tf.layers.flatten({ name: name + '_flatten' }).apply(x) as tf.SymbolicTensor;
}

export function applyCrossAttentionLayer(
    name: string,
    dModel: number, // 32 | 64 |....
    numHeads: number,
    {
        battleInput,
        tankInput,
        alliesInput,
        enemiesInput,
        bulletsInput,
        alliesMaskInput,
        enemiesMaskInput,
        bulletsMaskInput,
    }: ReturnType<typeof createInputs>,
) {
    name = name + '_applyCrossAttentionLayer';

    // ---------- embed everything to a common d_model ---------------------------
    const tankTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(tankInput, dModel, name + '_' + tankInput.name)) as tf.SymbolicTensor;
    const battleTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(battleInput, dModel, name + '_' + battleInput.name));
    const allyTok = proj(alliesInput, dModel, name + '_' + alliesInput.name);
    const enemyTok = proj(enemiesInput, dModel, name + '_' + enemiesInput.name);
    const bulletTok = proj(bulletsInput, dModel, name + '_' + bulletsInput.name);

    const kvTok = tf.layers.concatenate({ name: name + '_tokens', axis: 1 })
        .apply([battleTok, allyTok, enemyTok, bulletTok] as tf.SymbolicTensor[]) as tf.SymbolicTensor;            // [B,S,d]

    // ---------- build 0/1 padding mask -----------------------------------------
    const battleInputFixedMask = new OnesMask({ name: name + '_battleInputFixedMask' }).apply(battleInput) as tf.SymbolicTensor;   // [dModel,2]
    const maskKv = tf.layers.concatenate({ name: name + '_mask', axis: 1 })
        .apply([
            battleInputFixedMask,
            alliesMaskInput,
            enemiesMaskInput,
            bulletsMaskInput,
        ] as SymbolicTensor[]) as tf.SymbolicTensor;           // [B,S]

    console.assert(
        kvTok.shape[1] === maskKv.shape[1],
        `Mask length ${ maskKv.shape[1] } ≠ kvTok slots ${ kvTok.shape[1] }`,
    );

    // ---------- self-attention block -------------------------------------------
    let x = new CrossAttentionLayer({
        name: name + '_CrossAttentionLayer',
        numHeads: numHeads,
        keyDim: dModel / numHeads,
    }).apply([tankTok, kvTok, maskKv]) as tf.SymbolicTensor;

    x = tf.layers.add({ name: name + '_add' }).apply([x, tankTok]) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: name + '_normalization', epsilon: 1e-5 }).apply(x) as tf.SymbolicTensor;

    return tf.layers.flatten({ name: name + '_flatten' }).apply(x) as tf.SymbolicTensor;
}

