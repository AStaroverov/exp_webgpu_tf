import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { ceil } from '../../../../../lib/math.ts';
import { Model } from '../def.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../Layers/PerceiverLayer.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { applySelfTransformLayers, createNormalizationLayer } from '../ApplyLayers.ts';
import { createInputs, convertInputsToTokens } from '../Inputs.ts';
import { GRID_SIZE } from '../Create.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 128,
    heads: 4,
    depth: 1,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 2,
    depth: 0.5,
};


export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const summarize = ({
        name,
        heads,
        length,
        token,
        mask,
        perceiverDepth,
        transformerDepth,
    }: {
        name: string,
        heads: number,
        length: number,
        perceiverDepth: number,
        transformerDepth: number,
        token: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
        mask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    }) => {
        const latentQ = new VariableLayer({
            name: name + '_latentQ',
            shape: [length, config.dim],
            initializer: 'truncatedNormal',
        }).apply(tokens.tankTok) as tf.SymbolicTensor;

        const perceiver = applyPerceiverLayer({
            name: name + '_perceiverLayer',
            depth: perceiverDepth,
            heads,
            qTok: latentQ,
            kvTok: token,
            kvMask: mask,
            preNorm: true,
        });

        const transformer = transformerDepth > 0 ? applySelfTransformLayers(name + '_transformer', {
            heads,
            depth: transformerDepth,
            token: perceiver,
            preNorm: true,
        }) : perceiver;

        return transformer;
    }

    const getVehicleToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + '_vehicleToken' + i, axis: 1 })
            .apply([
                tokens.alliesTok,
                tokens.enemiesTok,
            ]) as tf.SymbolicTensor;
    }
    const getVehicleMask = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + '_vehicleMask' + i, axis: 1 })
            .apply([
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
            ]) as tf.SymbolicTensor;
    };
    const summarizedVehicle = summarize({
        name: modelName + '_summarizedVehicle',
        heads: config.heads,
        length: 6,
        token: getVehicleToken,
        mask: getVehicleMask,
        perceiverDepth: ceil(2 * config.depth),
        transformerDepth: ceil(2 * config.depth),
    });

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

    // Small transformer on each CNN output separately
    const summarizedGrid = applySelfTransformLayers(modelName + '_gridTransformer', {
        heads: config.heads,
        depth: ceil(1 * config.depth),
        token: gridFeatures,
        preNorm: true,
    }); // [B, 16, dim]

    const summarizedRays = applySelfTransformLayers(modelName + '_raysTransformer', {
        heads: config.heads,
        depth: ceil(1 * config.depth),
        token: raysFeatures,
        preNorm: true,
    }); // [B, 8, dim]

    // Projectiles Token
    const summarizedProjectiles = summarize({
        name: modelName + '_summarizedProjectiles',
        heads: config.heads,
        length: 2,
        token: tokens.bulletsTok,
        mask: inputs.bulletsMaskInput,
        perceiverDepth: ceil(1 * config.depth),
        transformerDepth: ceil(1 * config.depth),
    });

    // Heads Token
    const getHeadsToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + '_headsToken' + i, axis: 1 })
            .apply([
                tokens.tankTok,
                tokens.tankHistoryTok,
                tokens.turretTok,
                summarizedVehicle,
                summarizedGrid,
                summarizedRays,
                summarizedProjectiles,
            ]) as tf.SymbolicTensor;
    }
    const summarizedHeads = summarize({
        name: modelName + '_summarizedHeads',
        heads: config.heads,
        length: 4,
        token: getHeadsToken,
        perceiverDepth: ceil(4 * config.depth),
        transformerDepth: ceil(4 * config.depth),
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