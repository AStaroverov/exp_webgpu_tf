import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { Model } from '../../../../ppo/src/models/def.ts';
import { SliceLayer } from '../../../../ppo/src/models/Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../../../../ppo/src/models/Layers/PerceiverLayer.ts';
import { VariableLayer } from '../../../../ppo/src/models/Layers/VariableLayer.ts';
import { createNormalizationLayer } from '../../../../ppo/src/models/ApplyLayers.ts';
import { createInputs, convertInputsToTokens } from '../Inputs.ts';
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
    depth: 6,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 1,
    depth: 1,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);
  
    const getKvToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + 'kvToken' + i, axis: 1 })
            .apply([
                tokens.gridTok,
                tokens.raysTok,
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
                new MaskLikeLayer({ name: tokens.gridTok.name + '_maskLike' + i }).apply(tokens.gridTok) as tf.SymbolicTensor,
                new MaskLikeLayer({ name: tokens.raysTok.name + '_maskLike' + i }).apply(tokens.raysTok) as tf.SymbolicTensor,
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
        shape: [32, config.dim],
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
        shape: [4, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const summarizedHeads = applyPerceiverLayer({
        name: modelName + '_summarizedHeads',
        heads: 1,
        depth: 2,
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
