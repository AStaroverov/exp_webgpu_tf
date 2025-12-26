import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS, BULLET_FEATURES_DIM,
    BULLET_SLOTS, ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    ENV_RAY_FEATURES_DIM,
    ENV_RAY_SLOTS,
    TANK_FEATURES_DIM,
    TURRET_RAY_FEATURES_DIM,
    TURRET_RAY_SLOTS,
    RAY_HIT_TYPE_COUNT,
} from './Create.ts';
import { MaskLikeLayer } from './Layers/MaskLikeLayer.ts';
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';
import { RMSNormConfig, RMSNormLayer } from "./Layers/RMSNormLayer.ts";
import { VariableLayer } from './Layers/VariableLayer.ts';
import { NoisyDenseLayer } from './Layers/NoisyDenseLayer.ts';
import { VEHICLE_TYPE_COUNT } from '../../../tanks/src/Game/Config/vehicles.ts';

export function createInputs(name: string) {
    const tankInput = tf.input({name: name + '_tankInput', shape: [TANK_FEATURES_DIM]});
    const tankTypeInput = tf.input({name: name + '_tankTypeInput', shape: [1]});
    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesTypesInput = tf.input({name: name + '_enemiesTypesInput', shape: [ENEMY_SLOTS]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS]});
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesTypesInput = tf.input({name: name + '_alliesTypesInput', shape: [ALLY_SLOTS]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [ALLY_SLOTS]});
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS]});
    const envRaysInput = tf.input({name: name + '_envRaysInput', shape: [ENV_RAY_SLOTS, ENV_RAY_FEATURES_DIM]});
    const envRaysTypes = tf.input({name: name + '_envRaysTypes', shape: [ENV_RAY_SLOTS]});
    const turretRaysInput = tf.input({name: name + '_turretRaysInput', shape: [TURRET_RAY_SLOTS, TURRET_RAY_FEATURES_DIM]});
    const turretRaysTypes = tf.input({name: name + '_turretRaysTypes', shape: [TURRET_RAY_SLOTS]});

    return {
        tankInput,
        tankTypeInput,
        enemiesInput,
        enemiesTypesInput,
        enemiesMaskInput,
        alliesInput,
        alliesTypesInput,
        alliesMaskInput,
        bulletsInput,
        bulletsMaskInput,
        envRaysInput,
        envRaysTypes,
        turretRaysInput,
        turretRaysTypes,
    };
}

export function applyMLP({name, layers: hiddenLayers, preNorm = false}: {
    name: string,
    layers: [ActivationIdentifier, number][],
    preNorm?: boolean
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/MLP_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        layer = createDenseLayer({
            name: `${name}/MLP_dense${i++}`,
            units,
            activation,
            useBias: true
        }).apply(layer) as tf.SymbolicTensor;
    }

    return layer;
}

export function createDenseLayer(options: DenseLayerArgs & Required<Pick<DenseLayerArgs, 'useBias' | 'activation'>> & { noisy?: boolean, sigma?: number }) {
    return options.noisy
        ? new NoisyDenseLayer(options)
        : tf.layers.dense(options);
}

// https://arxiv.org/html/2406.09079v1
export function applyLaNLayer({name, units, preNorm = false}: {
    name: string,
    units: number,
    preNorm?: boolean,
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/LaN_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    const branch1 = createDenseLayer({
        name: `${name}/LaN_branch1`,
        units,
        useBias: true,
        activation: 'tanh',
    }).apply(layer) as tf.SymbolicTensor;
    const branch2 = createDenseLayer({
        name: `${name}/LaN_branch2`,
        units,
        useBias: true,
        activation: 'tanh',
    }).apply(layer) as tf.SymbolicTensor;

    return tf.layers.multiply({name: `${name}/LaN_output`}).apply([branch1, branch2]) as tf.SymbolicTensor;
}

export function applyNoisyLaNLayer({name, units, sigma, preNorm = false}: {
    name: string,
    units: number,
    sigma?: number,
    preNorm?: boolean,
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/NoisyLaN_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    const branch1 = new NoisyDenseLayer({
        name: `${name}/NoisyLaN_branch1`,
        units,
        useBias: true,
        activation: 'tanh',
        sigma,
    }).apply(layer) as tf.SymbolicTensor;
    const branch2 = new NoisyDenseLayer({
        name: `${name}/NoisyLaN_branch2`,
        units,
        useBias: true,
        activation: 'tanh',
        sigma,
    }).apply(layer) as tf.SymbolicTensor;

    return tf.layers.multiply({name: `${name}/NoisyLaN_output`}).apply([branch1, branch2]) as tf.SymbolicTensor;
}

export function createNormalizationLayer(options: RMSNormConfig) {
    return new RMSNormLayer(options);
}

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
   return createDenseLayer({
        name: name + '_tokProj',
        units: dModel,
        useBias: false,
        activation: 'linear'
    }).apply(x) as SymbolicTensor;
}

export function convertInputsToTokens(
    {
        tankInput,
        tankTypeInput,
        enemiesInput,
        enemiesTypesInput,
        alliesInput,
        alliesTypesInput,
        bulletsInput,
        envRaysInput,
        envRaysTypes,
        turretRaysInput,
        turretRaysTypes,
    }: ReturnType<typeof createInputs>,
    dModel: number,
) {
    const EMBEDDING_DIM = 3;

    // Utility: reshape 2D [batch, features] to 3D [batch, 1, features]
    const to3D = (x: tf.SymbolicTensor): tf.SymbolicTensor => {
        const lastDim = x.shape[x.shape.length - 1] as number;
        return tf.layers.reshape({
            name: `${x.name}_3d`,
            targetShape: [1, lastDim],
        }).apply(x) as tf.SymbolicTensor;
    };

    // Token type embedding: learnable vector broadcast to match input sequence length
    const applyTokenTypeEmbedding = (name: string, input: tf.SymbolicTensor): tf.SymbolicTensor => {
        const seqLen = input.shape[1] as number;
        const emb = new VariableLayer({
            name: `${name}_tokenTypeEmbedding`,
            shape: [EMBEDDING_DIM],
        }).apply(input) as tf.SymbolicTensor;
        return tf.layers.repeatVector({
            name: `${name}_tokenTypeEmb_repeat`,
            n: seqLen,
        }).apply(emb) as tf.SymbolicTensor;
    };

    // Project input concatenated with embeddings to dModel
    const projectWithEmbeddings = (
        name: string,
        input: tf.SymbolicTensor,
        ...typeEmbs: tf.SymbolicTensor[]
    ): tf.SymbolicTensor => {
        const parts = [input, ...typeEmbs];
        const concat = tf.layers.concatenate({ name: `${name}_concat`, axis: -1 })
            .apply(parts) as tf.SymbolicTensor;
            
        return createDenseLayer({
            name: `${name}_tokProj`,
            units: dModel,
            useBias: false,
            activation: 'linear',
        }).apply(concat) as tf.SymbolicTensor;
    };

    // Shared vehicle type embedding for tank, enemies, allies (small axial embedding)
    const vehicleTypeEmbedding = tf.layers.embedding({
        name: 'vehicleType_sharedEmbedding',
        inputDim: VEHICLE_TYPE_COUNT,
        outputDim: EMBEDDING_DIM,
        embeddingsInitializer: 'glorotNormal',
    });

    // Tank: expand to 3D [batch, 1, features], concat with embeddings, then project
    // All outputs must be 3D for transformer
    const tankInput3D = to3D(tankInput);
    const tankTokenTypeEmb = applyTokenTypeEmbedding('tank', tankInput3D);
    const tankVehicleEmb = vehicleTypeEmbedding.apply(tankTypeInput) as tf.SymbolicTensor;
    const tankTok = projectWithEmbeddings('tank', tankInput3D, tankTokenTypeEmb, tankVehicleEmb);

    // Enemies: concat input + token type emb + vehicle type emb, then project
    const enemiesTokenTypeEmb = applyTokenTypeEmbedding('enemies', enemiesInput);
    const enemiesVehicleEmb = vehicleTypeEmbedding.apply(enemiesTypesInput) as tf.SymbolicTensor;
    const enemiesTok = projectWithEmbeddings('enemies', enemiesInput, enemiesTokenTypeEmb, enemiesVehicleEmb);

    // Allies: concat input + token type emb + vehicle type emb, then project
    const alliesTokenTypeEmb = applyTokenTypeEmbedding('allies', alliesInput);
    const alliesVehicleEmb = vehicleTypeEmbedding.apply(alliesTypesInput) as tf.SymbolicTensor;
    const alliesTok = projectWithEmbeddings('allies', alliesInput, alliesTokenTypeEmb, alliesVehicleEmb);

    // Bullets: concat input + token type emb, then project (no categorical embedding)
    const bulletsTokenTypeEmb = applyTokenTypeEmbedding('bullets', bulletsInput);
    const bulletsTok = projectWithEmbeddings('bullets', bulletsInput, bulletsTokenTypeEmb);

    // Shared hit type embedding for all rays (small axial embedding)
    const rayHitTypeEmbedding = tf.layers.embedding({
        name: 'rayHitType_sharedEmbedding',
        inputDim: RAY_HIT_TYPE_COUNT,
        outputDim: EMBEDDING_DIM,
        embeddingsInitializer: 'glorotNormal',
    });

    // Environment rays: concat input + token type emb + hit type emb, then project
    const envRaysTokenTypeEmb = applyTokenTypeEmbedding('envRays', envRaysInput);
    const envRaysHitEmb = rayHitTypeEmbedding.apply(envRaysTypes) as tf.SymbolicTensor;
    const envRaysTok = projectWithEmbeddings('envRays', envRaysInput, envRaysTokenTypeEmb, envRaysHitEmb);

    // Turret rays: concat input + token type emb + hit type emb, then project
    const turretRaysTokenTypeEmb = applyTokenTypeEmbedding('turretRays', turretRaysInput);
    const turretRaysHitEmb = rayHitTypeEmbedding.apply(turretRaysTypes) as tf.SymbolicTensor;
    const turretRaysTok = projectWithEmbeddings('turretRays', turretRaysInput, turretRaysTokenTypeEmb, turretRaysHitEmb);

    return {
        tankTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
        envRaysTok,
        turretRaysTok,
    };
}

export function applyCrossAttentionLayer(
    {
        name,
        heads,
        qTok,
        kvTok,
        qMask,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
        qMask?: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;
    qTok = preNorm
        ? createNormalizationLayer({ name: name + '_QNorm_' + qTok.name}).apply(qTok) as tf.SymbolicTensor
        : qTok;
    kvTok = qTok === kvTok
        ? qTok
        : preNorm
            ? createNormalizationLayer({ name: name + '_KVNorm_' + kvTok.name}).apply(kvTok) as tf.SymbolicTensor
            : kvTok;

    // Create mask-like layers if masks are not provided
    qMask ??= new MaskLikeLayer({ name: qTok.name + '_qMaskLike' }).apply(qTok) as tf.SymbolicTensor;
    kvMask ??= new MaskLikeLayer({ name: kvTok.name + '_kvMaskLike' }).apply(kvTok) as tf.SymbolicTensor;

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / heads,
        numHeads: heads,
    }).apply([qTok, qMask, kvTok, kvMask]) as tf.SymbolicTensor;

    return attention;
}

export function applyCrossTransformerLayer(
    {
        name,
        heads,
        qTok,
        qMask,
        kvTok,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        qMask?: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;

    const crossAttn = applyCrossAttentionLayer({name, heads, qTok, qMask, kvTok, kvMask, preNorm});

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([qTok, crossAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ffnLN`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'relu',
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear'
    }).apply(ffnInner) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}

export function applyCrossTransformerLayers({
    name,
    depth,
    heads,
    qTok,
    kvTok,
    qMask,
    kvMask,
    preNorm = false,
}: {
    name: string, 
    depth: number,
    heads: number,
    qTok: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    kvTok: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    qMask?: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    kvMask?: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    preNorm?: boolean,
}) {
    let x = typeof qTok === 'function' ? qTok() : qTok;
    for (let i = 0; i < depth; i++) {
        x = applyCrossAttentionLayer({
            name: `${name}/depth${i}`,
            heads,
            qTok: x,
            kvTok: typeof kvTok === 'function' ? kvTok() : kvTok,
            qMask: typeof qMask === 'function' ? qMask() : qMask,
            kvMask: typeof kvMask === 'function' ? kvMask() : kvMask,
            preNorm,
        });
    }

    return x;
}

export function applySelfTransformerLayer(
    {
        name,
        heads,
        token,
        mask,
        noisy = false,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor;
        mask?: tf.SymbolicTensor;
        noisy?: boolean,
        preNorm?: boolean;
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;
    const selfAttn = applyCrossAttentionLayer({
        name,
        heads,
        qTok: token,
        qMask: mask,
        kvTok: token,
        kvMask: mask,
        preNorm
    });

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([token, selfAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ln2`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'relu',
        noisy,
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear',
        noisy,
    }).apply(ffnInner) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}

export function applySelfTransformLayers(name: string, {
    depth,
    heads,
    token,
    mask,
    noisy = false,
    preNorm = false,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    mask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    noisy?: boolean,
    preNorm?: boolean,
}) {
    let x = typeof token === 'function' ? token(name, 0) : token;
    for (let i = 0; i < depth; i++) {
        const lName = `${name}/depth${i}`;
        x = applySelfTransformerLayer({
            name: lName,
            heads,
            token: x,
            mask: mask ? (typeof mask === 'function' ? mask(lName, i) : mask) : undefined,
            noisy,
            preNorm,
        });
    }

    return x;
}

export function applyGlobalAverage1d({ name }: { name: string }, token: tf.SymbolicTensor) {
    return tf.layers.globalAveragePooling1d({ name: name + '_GlobalAvgPool1D' })
        .apply(token) as tf.SymbolicTensor;
}