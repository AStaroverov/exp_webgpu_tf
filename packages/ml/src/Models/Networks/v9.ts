import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../../lib/throwingError.ts';
import { ceil } from '../../../../../lib/math.ts';
import { Model } from '../def.ts';
import { MaskLikeLayer } from '../Layers/MaskLikeLayer.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';
import { applyPerceiverLayer } from '../Layers/PerceiverLayer.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { applyLaNLayer } from '../ApplyLayers.ts';
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
    depth: 6,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 1,
    depth: 3,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const getContextToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + '_contextToken' + i, axis: 1 })
            .apply([
                tokens.tankTok,
                tokens.turretTok,
                tokens.raysTok,
                tokens.alliesTok,
                tokens.enemiesTok,
                tokens.bulletsTok,
            ]) as tf.SymbolicTensor;
    }

    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const getContextMask = (name: string, i: number) => {
        const tankMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
        const turretMask = maskLike.apply(tokens.turretTok) as tf.SymbolicTensor;
        const raysMask = maskLike.apply(tokens.raysTok) as tf.SymbolicTensor;
        
        return tf.layers.concatenate({name: name + '_contextTokenMask' + i, axis: 1 })
            .apply([
                tankMask,
                turretMask,
                raysMask,
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
                inputs.bulletsMaskInput,
            ]) as tf.SymbolicTensor;
    };

    const latentQ = new VariableLayer({
        name: modelName + '_latentQ',
        shape: [12, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const latentPerceiver = applyPerceiverLayer({
        name: modelName + '_latentPerceiver',
        depth: ceil(config.depth * 0.66),
        heads: config.heads,
        qTok: latentQ,
        kvTok: getContextToken,
        kvMask: getContextMask,
        preNorm: true,
    });

    const headsQ = new VariableLayer({
        name: modelName + '_headsQ',
        shape: [4, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const headsPerceiver = applyPerceiverLayer({
        name: modelName + '_headsPerceiver',
        depth: ceil(config.depth * 0.33),
        heads: config.heads,
        qTok: headsQ, 
        kvTok: latentPerceiver,
        preNorm: true,
    });

    const finalToken = applyLaNLayer({
        name: modelName + '_finalToken',
        units: config.dim,
        preNorm: true
    }, headsPerceiver);

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
