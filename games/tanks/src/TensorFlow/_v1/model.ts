import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { INPUT_DIM } from '../Common/consts';
import { MODEL_PARAMS } from './const';
import { createInputVector } from '../Common/createInputVector.ts';

// Инициализация WASM бэкенда
export async function initWasmBackend() {
    await tf.setBackend('wasm');
    console.log('WASM backend initialized');
}

// Размерности выходных данных
const MOVEMENT_OUTPUT_DIM = 2;  // x, y вектор движения
const AIM_OUTPUT_DIM = 2;       // x, y координаты цели
const SHOOT_OUTPUT_DIM = 1;     // вероятность выстрела

// Тип буфера воспроизведения
export type ReplayBuffer = {
    states: tf.Tensor[],
    actions: {
        movement: tf.Tensor[],
        aim: tf.Tensor[],
        shoot: tf.Tensor[]
    },
    rewards: tf.Tensor[],
    nextStates: tf.Tensor[],
    done: tf.Tensor[]
};

// Тип предсказания модели
export type ModelPrediction = {
    movement: { x: number, y: number },
    aim: { x: number, y: number },
    shoot: boolean
};

// Создание архитектуры модели
function buildModel(): tf.LayersModel {
    // Входной слой
    const input = tf.layers.input({ shape: [INPUT_DIM] });

    // Общие слои (shared backbone)
    const shared = tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelInitializer: 'heNormal',
    }).apply(input);

    const shared2 = tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelInitializer: 'heNormal',
    }).apply(shared);

    // Ветвь для движения (движение танка)
    const movementBranch = tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelInitializer: 'heNormal',
    }).apply(shared2);

    const movementOutput = tf.layers.dense({
        units: MOVEMENT_OUTPUT_DIM,
        activation: 'tanh',  // диапазон [-1, 1] для векторов движения
        name: 'movement_output',
    }).apply(movementBranch);

    // Ветвь для прицеливания (куда наводить башню)
    const aimBranch = tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelInitializer: 'heNormal',
    }).apply(shared2);

    const aimOutput = tf.layers.dense({
        units: AIM_OUTPUT_DIM,
        activation: 'sigmoid',  // диапазон [0, 1] для нормализованных координат
        name: 'aim_output',
    }).apply(aimBranch);

    // Ветвь для принятия решения о стрельбе
    const shootBranch = tf.layers.dense({
        units: 32,
        activation: 'relu',
        kernelInitializer: 'heNormal',
    }).apply(shared2);

    const shootOutput = tf.layers.dense({
        units: SHOOT_OUTPUT_DIM,
        activation: 'sigmoid',  // вероятность выстрела [0, 1]
        name: 'shoot_output',
    }).apply(shootBranch);

    // Создание модели с тремя выходами
    return tf.model({
        inputs: input,
        outputs: [movementOutput, aimOutput, shootOutput] as SymbolicTensor[],
    });
}

// Создание нового буфера воспроизведения
export function createReplayBuffer(): ReplayBuffer {
    return {
        states: [],
        actions: {
            movement: [],
            aim: [],
            shoot: [],
        },
        rewards: [],
        nextStates: [],
        done: [],
    };
}

// Компиляция модели
export function compileModel(model: tf.LayersModel): tf.LayersModel {
    model.compile({
        optimizer: tf.train.adam(MODEL_PARAMS.LEARNING_RATE),
        loss: {
            movement_output: 'meanSquaredError',
            aim_output: 'meanSquaredError',
            shoot_output: 'binaryCrossentropy',
        },
        metrics: {
            movement_output: 'mae',
            aim_output: 'mae',
            shoot_output: 'accuracy',
        },
    });
    console.log('Model compiled');
    return model;
}

// Создание и компиляция модели
export function createModel(): tf.LayersModel {
    const model = buildModel();
    return compileModel(model);
}

// Предсказание действий для конкретного танка
export function predictActions(
    model: tf.LayersModel,
    tankEid: number,
    width: number,
    height: number,
    maxSpeed: number,
): ModelPrediction {
    // Получение входного вектора
    const inputVector = createInputVector(tankEid, width, height, maxSpeed);

    // Преобразование в тензор
    const inputTensor = tf.tensor2d([Array.from(inputVector)]);

    // Получение предсказаний
    const [movementTensor, aimTensor, shootTensor] = model.predict(inputTensor) as tf.Tensor[];

    // Конвертация в JavaScript массивы
    const movement = movementTensor.dataSync();
    const aim = aimTensor.dataSync();
    const shoot = shootTensor.dataSync()[0] > 0.5; // бинарное решение на основе вероятности

    // Освобождение ресурсов
    inputTensor.dispose();
    movementTensor.dispose();
    aimTensor.dispose();
    shootTensor.dispose();

    // Денормализация координат прицеливания
    const aimX = aim[0] * width;
    const aimY = aim[1] * height;

    return {
        movement: { x: movement[0], y: movement[1] },
        aim: { x: aimX, y: aimY },
        shoot,
    };
}

// Добавление опыта в буфер воспроизведения
export function addExperience(
    buffer: ReplayBuffer,
    state: Float32Array,
    action: { movement: number[], aim: number[], shoot: boolean },
    reward: number,
    nextState: Float32Array,
    done: boolean,
    bufferSize: number = MODEL_PARAMS.REPLAY_BUFFER_SIZE,
): void {
    // Преобразование в тензоры
    const stateTensor = tf.tensor2d([Array.from(state)]);
    const nextStateTensor = tf.tensor2d([Array.from(nextState)]);
    const movementTensor = tf.tensor1d(action.movement);
    const aimTensor = tf.tensor1d(action.aim);
    const shootTensor = tf.tensor1d([action.shoot ? 1 : 0]);
    const rewardTensor = tf.scalar(reward);
    const doneTensor = tf.scalar(done ? 1 : 0);

    // Добавление в буфер
    buffer.states.push(stateTensor);
    buffer.actions.movement.push(movementTensor);
    buffer.actions.aim.push(aimTensor);
    buffer.actions.shoot.push(shootTensor);
    buffer.rewards.push(rewardTensor);
    buffer.nextStates.push(nextStateTensor);
    buffer.done.push(doneTensor);

    // Ограничение размера буфера
    if (buffer.states.length > bufferSize) {
        // Удаление и освобождение самого старого опыта
        buffer.states.shift()?.dispose();
        buffer.actions.movement.shift()?.dispose();
        buffer.actions.aim.shift()?.dispose();
        buffer.actions.shoot.shift()?.dispose();
        buffer.rewards.shift()?.dispose();
        buffer.nextStates.shift()?.dispose();
        buffer.done.shift()?.dispose();
    }
}

// Обучение модели на основе опыта из буфера воспроизведения
export async function trainModelOnBatch(
    model: tf.LayersModel,
    buffer: ReplayBuffer,
    batchSize: number = MODEL_PARAMS.BATCH_SIZE,
): Promise<boolean> {
    if (buffer.states.length < batchSize) {
        console.log('Not enough samples in replay buffer');
        return false;
    }

    // Выбор случайной партии из буфера
    const indices: number[] = [];
    for (let i = 0; i < batchSize; i++) {
        indices.push(Math.floor(Math.random() * buffer.states.length));
    }

    // Сбор тензоров для обучения
    const batchStates = tf.concat(indices.map(i => buffer.states[i]));
    const batchNextStates = tf.concat(indices.map(i => buffer.nextStates[i]));
    const batchMovementActions = tf.concat(indices.map(i => buffer.actions.movement[i]));
    const batchAimActions = tf.concat(indices.map(i => buffer.actions.aim[i]));
    const batchShootActions = tf.concat(indices.map(i => buffer.actions.shoot[i]));
    const batchRewards = tf.stack(indices.map(i => buffer.rewards[i]));
    const batchDone = tf.stack(indices.map(i => buffer.done[i]));

    try {
        // Обучение модели
        await model.trainOnBatch(
            batchStates,
            [batchMovementActions, batchAimActions, batchShootActions],
        );

        return true;
    } catch (error) {
        console.error('Error during training:', error);
        return false;
    } finally {
        // Освобождение ресурсов
        batchStates.dispose();
        batchNextStates.dispose();
        batchMovementActions.dispose();
        batchAimActions.dispose();
        batchShootActions.dispose();
        batchRewards.dispose();
        batchDone.dispose();
    }
}

// Сохранение модели в IndexedDB
export async function saveModel(model: tf.LayersModel, path: string): Promise<boolean> {
    try {
        await model.save(`indexeddb://${ path }`);
        console.log(`Model saved to indexeddb://${ path }`);
        return true;
    } catch (error) {
        console.error('Failed to save model:', error);
        return false;
    }
}

// Загрузка модели из IndexedDB
export async function loadModel(path: string): Promise<tf.LayersModel | null> {
    try {
        const model = await tf.loadLayersModel(`indexeddb://${ path }`);
        compileModel(model);
        console.log(`Model loaded from indexeddb://${ path }`);
        return model;
    } catch (error) {
        console.error('Failed to load model:', error);
        return null;
    }
}

// Очистка ресурсов модели и буфера
export function disposeResources(model: tf.LayersModel, buffer: ReplayBuffer): void {
    model.dispose();

    // Очистка буфера
    buffer.states.forEach(tensor => tensor?.dispose());
    buffer.actions.movement.forEach(tensor => tensor?.dispose());
    buffer.actions.aim.forEach(tensor => tensor?.dispose());
    buffer.actions.shoot.forEach(tensor => tensor?.dispose());
    buffer.rewards.forEach(tensor => tensor?.dispose());
    buffer.nextStates.forEach(tensor => tensor?.dispose());
    buffer.done.forEach(tensor => tensor?.dispose());
}