import * as tf from '@tensorflow/tfjs';
import { AdamOptimizer } from '@tensorflow/tfjs';

export async function setModelState(targetModel: tf.LayersModel, sourceModel: tf.LayersModel): Promise<tf.LayersModel> {
    const sourceWeights = sourceModel.getWeights();
    const targetWeights = targetModel.getWeights();

    const shapesMatch = sourceWeights.length === targetWeights.length &&
        sourceWeights.every((w, i) => tf.util.arraysEqual(w.shape, targetWeights[i].shape));

    if (!shapesMatch) {
        throw new Error('Weight shapes do not match between models.');
    }

    targetModel.setWeights(sourceWeights);

    const sourceOptimizer = sourceModel.optimizer as undefined | AdamOptimizer;
    const targetOptimizer = targetModel.optimizer as undefined | AdamOptimizer;

    if (sourceOptimizer && targetOptimizer) {
        const targetWeights = await targetOptimizer.getWeights();
        const sourceWeights = await sourceOptimizer.getWeights();
        await targetOptimizer.setWeights(sourceWeights);
        tf.dispose(targetWeights.map(nt => nt.tensor));
        tf.dispose(sourceWeights.map(nt => nt.tensor));
        // @ts-ignore
        targetOptimizer.learningRate = sourceOptimizer.learningRate;
        // @ts-ignore
        targetOptimizer.beta1 = sourceOptimizer.beta1;
        // @ts-ignore
        targetOptimizer.beta2 = sourceOptimizer.beta2;
        // @ts-ignore
        targetOptimizer.epsilon = sourceOptimizer.epsilon;
    }

    if (sourceModel.loss != null) {
        targetModel.loss = sourceModel.loss;
    }
    if (sourceModel.metrics != null) {
        targetModel.metrics = sourceModel.metrics;
    }

    return targetModel;
}