import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { Model } from '../../../../ppo/src/models/def.ts';
import { SliceLayer } from '../../../../ppo/src/models/Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../../../../ppo/src/models/Layers/PerceiverLayer.ts';
import { VariableLayer } from '../../../../ppo/src/models/Layers/VariableLayer.ts';
import { createNormalizationLayer } from '../../../../ppo/src/models/ApplyLayers.ts';
import { createInputs, convertInputsToTokens } from '../Inputs.ts';
import { GRID_SIZE } from '../dims.ts';
import { MaskLikeLayer } from '../../../../ppo/src/models/Layers/MaskLikeLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 128,
    heads: 4,
    depth: 3,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 1,
    depth: 1,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    // Spatial CNN compression: grid (Conv2D) + rays (Conv1D)
    const gridFeatures = applyGridCNN({
        name: modelName + '_gridCNN',
        input: inputs.obstacleGridInput,
        dim: config.dim,
        spatialShape: [GRID_SIZE, GRID_SIZE],
        steps: 2,
        poolSize: 2,
    }); // [B, 16, dim]

    const raysFeatures = applyCNN({
        name: modelName + '_raysCNN',
        input: inputs.raysInput,
        dim: config.dim,
        steps: 3,
        poolSize: 2,
    }); // [B, 16, dim]

    const getEntities = (i: number) => {
        return [
            [gridFeatures, new MaskLikeLayer({ name: gridFeatures.name + '_maskLike' + i }).apply(gridFeatures) as tf.SymbolicTensor] as const,
            [raysFeatures, new MaskLikeLayer({ name: raysFeatures.name + '_maskLike' + i }).apply(raysFeatures) as tf.SymbolicTensor] as const,
            [tokens.bulletsTok, inputs.bulletsMaskInput] as const,
            [tokens.alliesTok, inputs.alliesMaskInput] as const,
            [tokens.enemiesTok, inputs.enemiesMaskInput] as const,
            [tokens.tankHistoryTok, new MaskLikeLayer({ name: tokens.tankHistoryTok.name + '_maskLike' + i }).apply(tokens.tankHistoryTok) as tf.SymbolicTensor] as const,
            [tokens.tankTok, new MaskLikeLayer({ name: tokens.tankTok.name + '_maskLike' + i }).apply(tokens.tankTok) as tf.SymbolicTensor] as const,
            [tokens.turretTok, new MaskLikeLayer({ name: tokens.turretTok.name + '_maskLike' + i }).apply(tokens.turretTok) as tf.SymbolicTensor] as const,
        ] as const;
    }

    let latent = new VariableLayer({
        name: modelName + '_latent',
        shape: [16, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    for (let i = 0; i < config.depth; i++) {
        const entities = getEntities(i);
        for (const [token, mask] of entities) {
            latent = applyPerceiverLayer({
                name: token.name + '_perceiverLayer_' + i,
                depth: 1,
                heads: config.heads,
                qTok: latent,
                kvTok: token,
                kvMask: mask,
                preNorm: true,
            });
        }
    }

    const latentHeads = new VariableLayer({
        name: modelName + '_heads',
        shape: [4, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const summarizedHeads = applyPerceiverLayer({
        name: modelName + '_summarizedHeads',
        heads: config.heads,
        depth: 1,
        qTok: latentHeads,
        kvTok: latent
    });

    const finalToken = createNormalizationLayer({name: modelName + '_finalToken'}).apply(summarizedHeads) as tf.SymbolicTensor;

    if (modelName === Model.Policy) {
        const heads = Array.from({ length: 4 }, (_, i) => {
            const token = new SliceLayer({
                name: modelName + '_headToken' + i,
                beginSlice: [0, i, 0],
                sliceSize: [-1, 1, -1],
            }).apply(finalToken) as tf.SymbolicTensor;
            return tf.layers
                .flatten({ name: modelName + '_flattenedHeadToken' + i })
                .apply(token) as tf.SymbolicTensor;
        });
        return { inputs, heads };
    }

     if (modelName === Model.Value) {
        const averageToken = tf.layers
            .globalAveragePooling1d({ name: modelName + '_averageToken' })
            .apply(finalToken) as tf.SymbolicTensor;
        return { inputs, heads: [averageToken] };
     }

     throwingError(`Invalid model name: ${modelName}`);
}

/**
 * 1D CNN compression: Conv1D + MaxPool1D.
 * Output: [B, L / poolSize^steps, dim]
 * Filters scale per step: dim/2^(steps-1), ..., dim/2, dim
 */
function applyCNN({
    name,
    input,
    dim,
    steps,
    poolSize,
}: {
    name: string;
    input: tf.SymbolicTensor;
    dim: number;
    steps: number;
    poolSize: number;
}): tf.SymbolicTensor {
    let x: tf.SymbolicTensor = input;

    for (let i = 0; i < steps; i++) {
        const filters = dim >> (steps - 1 - i);
        x = tf.layers.conv1d({ name: `${name}_conv${i}`, filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.maxPooling1d({ name: `${name}_pool${i}`, poolSize }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}

/**
 * 2D CNN compression for grid data.
 * Reshapes [B, H*W, F] → [B, H, W, F], applies Conv2D + MaxPool2D, reshapes back to [B, tokens, dim].
 */
function applyGridCNN({
    name,
    input,
    dim,
    spatialShape,
    steps,
    poolSize,
}: {
    name: string;
    input: tf.SymbolicTensor;
    dim: number;
    spatialShape: [number, number];
    steps: number;
    poolSize: number;
}): tf.SymbolicTensor {
    const [h, w] = spatialShape;

    let x = tf.layers.reshape({
        name: name + '_reshape2d',
        targetShape: [h, w, input.shape[input.shape.length - 1]!],
    }).apply(input) as tf.SymbolicTensor;

    for (let i = 0; i < steps; i++) {
        const filters = dim >> (steps - 1 - i);
        x = tf.layers.conv2d({ name: `${name}_conv${i}`, filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.maxPooling2d({ name: `${name}_pool${i}`, poolSize }).apply(x) as tf.SymbolicTensor;
    }

    const scale = poolSize ** steps;
    const outTokens = (h / scale) * (w / scale);
    x = tf.layers.reshape({
        name: name + '_reshape1d',
        targetShape: [outTokens, dim],
    }).apply(x) as tf.SymbolicTensor;

    return x;
}