import * as tf from '@tensorflow/tfjs';
import { ACTION_DIM, INPUT_DIM } from './consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export function createPolicyNetwork(hiddenLayers: [ActivationIdentifier, number][]): tf.LayersModel {
    // Входной тензор
    const input = tf.layers.input({
        name: 'policy/input',
        shape: [INPUT_DIM],
    });

    let x = input;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `policy/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    // Выход: ACTION_DIM * 2 нейронов (ACTION_DIM для mean, ACTION_DIM для logStd).
    // При использовании:
    //   mean = tanh(первые ACTION_DIM),
    //   std  = exp(последние ACTION_DIM).
    const policyOutput = tf.layers.dense({
        name: 'policy/output',
        units: ACTION_DIM * 2,
        activation: 'linear', // без ограничений, трансформации — вручную (tanh/exp)
    }).apply(x) as tf.SymbolicTensor;

    // Создаём модель
    return tf.model({
        name: 'policy',
        inputs: input,
        outputs: policyOutput,
    });
}

// Создание сети критика (оценки состояний)
export function createValueNetwork(hiddenLayers: [ActivationIdentifier, number][]): tf.LayersModel {
    // Входной слой
    const input = tf.layers.input({
        name: 'value/input',
        shape: [INPUT_DIM],
    });

    // Скрытые слои
    let x = input;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `value/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    // Выходной слой - скалярная оценка состояния
    const valueOutput = tf.layers.dense({
        name: 'value/output',
        units: 1,
        activation: 'linear',
    }).apply(x) as tf.SymbolicTensor;

    return tf.model({
        name: 'value',
        inputs: input,
        outputs: valueOutput,
    });
}