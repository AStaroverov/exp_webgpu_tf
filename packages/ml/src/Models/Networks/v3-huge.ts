import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';
import { createNetwork as createNetworkV3 } from './v3.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
    MLP: [ActivationIdentifier, number][];
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 128,
    heads: 4,
    depth: 6,
    MLP: [
        ['relu', 512],
        ['relu', 512],
        ['relu', 512],
        ['relu', 256],
        ['relu', 128],
        ['relu', 64],
    ],
};

const valueNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 2,
    depth: 2,
    MLP: [
        ['relu', 256],
        ['relu', 128],
        ['relu', 64],
        ['relu', 32],
    ] as [ActivationIdentifier, number][],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    return createNetworkV3(modelName, config);
}