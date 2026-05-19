import * as tf from '@tensorflow/tfjs';
import { ModelSettings } from '../core/types.ts';

export function setNetworkSettings(o: tf.LayersModel, settings: ModelSettings) {
    const meta = o.getUserDefinedMetadata() as ModelSettings;
    o.setUserDefinedMetadata({ ...meta, ...settings });
    setNetworkLearningRate(o, settings.lr ?? getNetworkLearningRate(o));
}

export function getNetworkSettings(network: tf.LayersModel): ModelSettings {
    const meta = network.getUserDefinedMetadata() as undefined | ModelSettings;
    return meta ?? {};
}

export function getNetworkExpIteration(network: tf.LayersModel): number {
    const meta = network.getUserDefinedMetadata() as undefined | { expIteration?: number };
    return meta?.expIteration ?? 0;
}

export function setNetworkLearningRate(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    const lr = network.optimizer?.learningRate;
    if (lr == null) {
        throw new Error('getNetworkLearningRate: optimizer.learningRate is not set on the network');
    }
    return lr;
}
