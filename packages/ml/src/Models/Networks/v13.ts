import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { Model } from '../def.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../Layers/PerceiverLayer.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { createNormalizationLayer } from '../ApplyLayers.ts';
import { createInputs, convertInputsToTokens } from '../Inputs.ts';
import { MaskLikeLayer } from '../Layers/MaskLikeLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 128,
    heads: 4,
    depth: 2,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 2,
    depth: 1,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);
  
    let gridLatentToken = new VariableLayer({
        name: modelName + '_gridLatent',
        shape: [16, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    gridLatentToken = applyPerceiverLayer({
        name: modelName + '_gridPerceiver',
        depth: 1,
        heads: config.heads,
        qTok: gridLatentToken,
        kvTok: tokens.gridTok,
        preNorm: true,
    });
    let raysLatentToken = new VariableLayer({
        name: modelName + '_raysLatent',
        shape: [16, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    raysLatentToken = applyPerceiverLayer({
        name: modelName + '_raysPerceiver',
        depth: 1,
        heads: config.heads,
        qTok: raysLatentToken,
        kvTok: tokens.raysTok,
        preNorm: true,
    });

      const getKvToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + 'kvToken' + i, axis: 1 })
            .apply([
                gridLatentToken,
                raysLatentToken,
                tokens.bulletsTok,
                tokens.alliesTok,
                tokens.enemiesTok,
                tokens.tankHistoryTok,
                tokens.tankTok,
                tokens.turretTok,
            ]) as tf.SymbolicTensor;
    }
    const getKvMask = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + 'kvMask' + i, axis: 1 })
            .apply([
                new MaskLikeLayer({ name: gridLatentToken.name + '_maskLike' + i }).apply(gridLatentToken) as tf.SymbolicTensor,
                new MaskLikeLayer({ name: raysLatentToken.name + '_maskLike' + i }).apply(raysLatentToken) as tf.SymbolicTensor,
                inputs.bulletsMaskInput,
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
                new MaskLikeLayer({ name: tokens.tankHistoryTok.name + '_maskLike' + i }).apply(tokens.tankHistoryTok) as tf.SymbolicTensor,
                new MaskLikeLayer({ name: tokens.tankTok.name + '_maskLike' + i }).apply(tokens.tankTok) as tf.SymbolicTensor,
                new MaskLikeLayer({ name: tokens.turretTok.name + '_maskLike' + i }).apply(tokens.turretTok) as tf.SymbolicTensor,            
            ]) as tf.SymbolicTensor;
    }

    const latentToken = new VariableLayer({
        name: modelName + '_latent',
        shape: [16, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const latentFeatures = applyPerceiverLayer({
        name: modelName + '_latentPerceiver',
        depth: config.depth,
        heads: config.heads,
        qTok: latentToken,
        kvTok: getKvToken( modelName + '_latentPerceiver', 0),
        kvMask: getKvMask( modelName + '_latentPerceiver', 0),
        preNorm: true,
    });

    const latentHeads = new VariableLayer({
        name: modelName + '_heads',
        shape: [config.heads, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const summarizedHeads = applyPerceiverLayer({
        name: modelName + '_summarizedHeads',
        heads: 1,
        depth: 1,
        qTok: latentHeads,
        kvTok: latentFeatures,
        preNorm: true,
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
