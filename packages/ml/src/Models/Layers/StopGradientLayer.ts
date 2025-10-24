import * as tf from '@tensorflow/tfjs';

const stopGradient = tf.customGrad((x) => {
    const gradFunc = () => tf.zerosLike(x as tf.Tensor);
    return { value: x as tf.Tensor, gradFunc };
});


export class StopGradientLayer extends tf.layers.Layer {
    static className = 'StopGradientLayer';

    call(inputs: tf.Tensor | tf.Tensor[]) {
        if (Array.isArray(inputs)) {
            return inputs.map(x => stopGradient(x));
        } else {
            return stopGradient(inputs);
        }
    }
}

tf.serialization.registerClass(StopGradientLayer);

