// Create Actor (Policy) Network with LSTM layer for better handling of partial observations
import * as tf from '@tensorflow/tfjs';
import { layers, LayersModel, sequential } from '@tensorflow/tfjs';
import { ACTION_DIM, INPUT_DIM } from './consts.ts';

export function createActorModel(): { meanModel: LayersModel, stdModel: LayersModel } {
    // Mean model with LSTM
    const meanModel = sequential();
    meanModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Reshape for LSTM
    meanModel.add(layers.reshape({ targetShape: [1, 128] }));

    // LSTM layer for sequence modeling
    meanModel.add(layers.lstm({
        units: 64,
        returnSequences: false,
        kernelInitializer: 'glorotNormal',
    }));

    meanModel.add(layers.dense({
        units: ACTION_DIM,
        activation: 'tanh',  // Using tanh for [-1, 1] range
        kernelInitializer: 'glorotNormal',
    }));

    // Std model (log standard deviations)
    const stdModel = sequential();
    stdModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 64,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    stdModel.add(layers.dense({
        units: ACTION_DIM,
        // Используем softplus вместо tanh для более мягкого ограничения логарифма стандартного отклонения
        activation: 'softplus',
        biasInitializer: tf.initializers.constant({ value: -1.0 }), // Начинаем с меньшего стандартного отклонения
    }));

    // Compile the models individually
    meanModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    stdModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return { meanModel, stdModel };
}

// Create Critic (Value) Network - также с LSTM
export function createCriticModel(): LayersModel {
    const model = sequential();

    // Input layer
    model.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Reshape for LSTM
    model.add(layers.reshape({ targetShape: [1, 128] }));

    // LSTM layer for sequence modeling
    model.add(layers.lstm({
        units: 64,
        returnSequences: false,
        kernelInitializer: 'glorotNormal',
    }));

    // Output layer - single value estimate
    model.add(layers.dense({
        units: 1,
        kernelInitializer: 'glorotNormal',
    }));

    model.compile({
        optimizer: tf.train.adam(0.0001), // Меньшая скорость обучения
        loss: 'meanSquaredError',
    });

    return model;
}