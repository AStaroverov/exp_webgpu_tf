import * as tf from '@tensorflow/tfjs';
import { AdamOptimizer } from '@tensorflow/tfjs';

export async function setModelState(targetModel: tf.LayersModel, sourceModel: tf.LayersModel): Promise<tf.LayersModel> {
    // 1. Копируем веса
    const sourceWeights = sourceModel.getWeights();
    const targetWeights = targetModel.getWeights();

    const shapesMatch = sourceWeights.length === targetWeights.length &&
        sourceWeights.every((w, i) => tf.util.arraysEqual(w.shape, targetWeights[i].shape));

    if (!shapesMatch) {
        throw new Error('Weight shapes do not match between models.');
    }

    targetModel.setWeights(sourceWeights);
    // 2. Копируем оптимайзер
    const sourceOptimizer = sourceModel.optimizer as undefined | AdamOptimizer;
    const targetOptimizer = targetModel.optimizer as undefined | AdamOptimizer;

    if (sourceOptimizer && targetOptimizer) {
        await targetOptimizer.setWeights(await sourceOptimizer.getWeights());
        // @ts-ignore
        targetOptimizer.learningRate = sourceOptimizer.learningRate;
        // @ts-ignore
        targetOptimizer.beta1 = sourceOptimizer.beta1;
        // @ts-ignore
        targetOptimizer.beta2 = sourceOptimizer.beta2;
        // @ts-ignore
        targetOptimizer.epsilon = sourceOptimizer.epsilon;
    }

    // 4. (необязательно) Копируем loss и metrics
    if (sourceModel.loss != null) {
        targetModel.loss = sourceModel.loss;
    }
    if (sourceModel.metrics != null) {
        targetModel.metrics = sourceModel.metrics;
    }

    return targetModel;
}