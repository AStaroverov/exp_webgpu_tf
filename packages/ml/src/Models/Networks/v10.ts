import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { ceil } from '../../../../../lib/math.ts';
import { Model } from '../def.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../Layers/PerceiverLayer.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { applySelfTransformLayers, createNormalizationLayer } from '../ApplyLayers.ts';
import { createInputs, convertInputsToTokens } from '../Inputs.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 64,
    heads: 4,
    depth: 1,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 1,
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
        length: 4,
        token: getVehicleToken,
        mask: getVehicleMask,
        perceiverDepth: ceil(3 * config.depth),
        transformerDepth: ceil(3 * config.depth),
    });

    // Rays Token
    const summarizedRays = summarize({
        name: modelName + '_summarizedRays',
        heads: config.heads,
        length: 4,
        token: tokens.raysTok,
        perceiverDepth: ceil(4 * config.depth),
        transformerDepth: ceil(4 * config.depth),
    });

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
                tokens.turretTok,
                summarizedVehicle,
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
    // applyLaNLayer({
    //     name: modelName + '_finalToken',
    //     units: config.dim,
    //     preNorm: true
    // }, summarizedHeads);

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
