import * as tf from '@tensorflow/tfjs';

const stopGradient = tf.customGrad((x) => {
    const gradFunc = () => tf.zerosLike(x as tf.Tensor);
    return { value: x as tf.Tensor, gradFunc };
});


export class StopGradientLayer extends tf.layers.Layer {
    static className = 'StopGradientLayer';
    call(inputs: tf.Tensor | tf.Tensor[]) {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        debugger
        return stopGradient(x as tf.Tensor);
    }
}

tf.serialization.registerClass(StopGradientLayer);

