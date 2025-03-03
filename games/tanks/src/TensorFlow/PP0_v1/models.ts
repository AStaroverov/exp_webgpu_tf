// Improved models.ts with optimized network architecture
import * as tf from '@tensorflow/tfjs';
import { layers, LayersModel, sequential } from '@tensorflow/tfjs';
import { ACTION_DIM, INPUT_DIM } from '../Common/consts.ts';
import { Regularizer, RegularizerIdentifier } from '@tensorflow/tfjs-layers/dist/regularizers';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

// Helper function to create a dense block with batch normalization and optional dropout
function createDenseBlock(
    model: tf.Sequential,
    units: number,
    activation: ActivationIdentifier = 'relu',
    dropoutRate: number = 0,
    kernelRegularizer?: Regularizer | RegularizerIdentifier,
) {
    model.add(layers.dense({
        units,
        activation,
        kernelInitializer: 'glorotNormal',
        kernelRegularizer,
    }));

    model.add(layers.batchNormalization());

    if (dropoutRate > 0) {
        model.add(layers.dropout({ rate: dropoutRate }));
    }

    return model;
}

export function createActorModel(): { meanModel: LayersModel, stdModel: LayersModel } {
    // L2 regularization to prevent overfitting
    const l2Regularizer = tf.regularizers.l2({ l2: 0.001 });

    // Mean model with wider layers and no LSTM
    const meanModel = sequential();

    // Input layer
    meanModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 256,  // Wider first layer
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: l2Regularizer,
    }));

    // Add batch normalization after input
    meanModel.add(layers.batchNormalization());

    // First hidden layer
    createDenseBlock(meanModel, 192, 'relu', 0.1, l2Regularizer);

    // Second hidden layer
    createDenseBlock(meanModel, 128, 'relu', 0.1, l2Regularizer);

    // Output layer for action means with tanh activation
    meanModel.add(layers.dense({
        units: ACTION_DIM,
        activation: 'tanh',  // For [-1, 1] range
        kernelInitializer: 'glorotNormal',
        // No regularization on output layer
        biasInitializer: tf.initializers.zeros(),
    }));

    // Standard deviation model
    const stdModel = sequential();

    // Input layer for std model
    stdModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: l2Regularizer,
    }));

    // Add batch normalization
    stdModel.add(layers.batchNormalization());

    // Hidden layer with less dropout
    createDenseBlock(stdModel, 64, 'relu', 0.05, l2Regularizer);

    // Output layer for standard deviations
    // Using softplus activation to ensure positive values
    stdModel.add(layers.dense({
        units: ACTION_DIM,
        activation: 'softplus',
        kernelInitializer: 'glorotNormal',
        // Start with smaller standard deviations to encourage exploitation initially
        biasInitializer: tf.initializers.constant({ value: -1.5 }),
    }));

    // Compile the models with Adam optimizer
    meanModel.compile({
        optimizer: tf.train.adam(0.0003),
        loss: 'meanSquaredError',
    });

    stdModel.compile({
        optimizer: tf.train.adam(0.0003),
        loss: 'meanSquaredError',
    });

    return { meanModel, stdModel };
}

// Create Critic (Value) Network - более широкая и глубокая сеть
export function createCriticModel(): LayersModel {
    const l2Regularizer = tf.regularizers.l2({ l2: 0.001 });
    const model = sequential();

    // Input layer - wider
    model.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 256,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: l2Regularizer,
    }));

    model.add(layers.batchNormalization());

    // First hidden layer
    createDenseBlock(model, 192, 'relu', 0.1, l2Regularizer);

    // Second hidden layer
    createDenseBlock(model, 128, 'relu', 0.1, l2Regularizer);

    // Third hidden layer
    createDenseBlock(model, 64, 'relu', 0.1, l2Regularizer);

    // Output layer - single value
    model.add(layers.dense({
        units: 1,
        kernelInitializer: 'glorotNormal',
        // No activation for unbounded value prediction
    }));

    // Use a smaller learning rate for value function to prevent oscillations
    model.compile({
        optimizer: tf.train.adam(0.0002),
        loss: 'meanSquaredError',
    });

    return model;
}

// New helper function to create exploration-biased actor models
// Useful for initializing agents that explore more at the beginning
export function createExplorationBiasedActorModel(): { meanModel: LayersModel, stdModel: LayersModel } {
    const { meanModel, stdModel } = createActorModel();

    // Modify the std model to have higher initial values
    // This will encourage more exploration
    const lastLayer = stdModel.layers[stdModel.layers.length - 1] as tf.layers.Layer;

    // Set bias to higher values
    const originalBias = lastLayer.getWeights()[1];
    const initialBiasValues = tf.ones(originalBias.shape).mul(tf.scalar(-0.5));

    // Apply the new biases
    lastLayer.setWeights([
        lastLayer.getWeights()[0],  // Keep the original weights
        initialBiasValues,           // Set higher initial standard deviations
    ]);

    return { meanModel, stdModel };
}

// Function to partially reset weights to help escape local minima
export function resetPartialWeights(model: LayersModel, resetProbability: number): void {
    // For each layer (except the last layer)
    for (let i = 0; i < model.layers.length - 1; i++) {
        const layer = model.layers[i];

        // Only reset weights for trainable dense layers
        if (layer.getClassName() === 'Dense' && layer.trainable) {
            const weights = layer.getWeights();

            // Only reset kernel weights, not biases
            const kernel = weights[0];
            const bias = weights[1];

            // Create a random mask of the same shape
            const resetMask = tf.tidy(() => {
                const random = tf.randomUniform(kernel.shape, 0, 1);
                return random.less(tf.scalar(resetProbability));
            });

            // Reset selected weights
            const newKernel = tf.tidy(() => {
                const newRandomWeights = tf.randomNormal(
                    kernel.shape,
                    0,
                    0.1,  // Small standard deviation for stability
                );

                return tf.where(
                    resetMask,
                    newRandomWeights,  // Replace with new values
                    kernel,             // Keep existing values
                );
            });

            // Apply the new weights
            layer.setWeights([newKernel, bias]);

            // Clean up tensors
            resetMask.dispose();
            newKernel.dispose();
        }
    }
}