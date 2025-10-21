import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';

export type PolicyOutput = {
    mean: tf.Tensor2D;
    phi?: tf.Tensor2D; // gSDE features (если gSDE включён)
};

export function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[]): PolicyOutput {
    if (CONFIG.gSDE.enabled) {
        // gSDE включён - ожидаем массив [mean, phi]
        if (!Array.isArray(prediction)) {
            throw new Error('Expected array output from policy network with gSDE enabled');
        }

        const [mean, phi] = prediction as [tf.Tensor2D, tf.Tensor2D];
        return { mean, phi };
    } else {
        // gSDE выключен - только mean
        if (Array.isArray(prediction)) {
            throw new Error('Expected single tensor output from policy network with gSDE disabled');
        }

        return { mean: prediction as tf.Tensor2D };
    }
}
