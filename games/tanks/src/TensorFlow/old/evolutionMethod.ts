import * as tf from '@tensorflow/tfjs';
import { layers, LayersModel, sequential, Sequential, tensor } from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { Tank } from '../../ECS/Components/Tank.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { inRange } from 'lodash-es';
import { createBattlefield } from '../createBattlefield.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { dist2, max } from '../../../../../lib/math.ts';
import { query } from 'bitecs';

setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
await tf.setBackend('wasm');

export const TANK_COUNT_SIMULATION = 6;

const TICK_TIME_REAL = 1;
const TICK_TIME_SIMULATION = 16.6667; // 60 FPS
const POPULATION_SIZE = TANK_COUNT_SIMULATION * 10;
const GENERATIONS = 500;
const ELITE_COUNT = 3; // число лучших особей, сохраняемых без изменений
const MUTATION_RATE = 0.1; // вероятность мутации каждого веса
const SIMULATION_STEPS = 1000; // число шагов симуляции (можно увеличить для длительной симуляции)

const reward = {
    aim: 0,
    bulletAvoidance: 0,
    mapStay: 0,
    health: 0,
};
const resetReward = () => {
    reward.aim = 0;
    reward.bulletAvoidance = 0;
    reward.mapStay = 0;
    reward.health = 0;
};

// Функция симуляции игры для N танков
// Здесь один танк ассоциирован с одним противником (если противников меньше, можно добавить padding или повторять существующих)
async function simulateGame(models: tf.LayersModel[]): Promise<number[]> {
    return new Promise(resolve => {
        const { world, canvas, gameTick, destroy } = createBattlefield(TANK_COUNT_SIMULATION);

        const mapTankToModel = new Map<number, LayersModel>();
        const mapTankToOutputTensor = new Map<number, tf.Tensor>();
        const mapModelToFitness = new Map<LayersModel, number>();

        let steps = SIMULATION_STEPS;
        const stopInterval = macroTasks.addInterval(() => {
            gameTick(TICK_TIME_SIMULATION);

            const tankInputTensorEids = query(world, [Tank, TankController, TankInputTensor]);
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;

            if (tankInputTensorEids.length === 1) {
                const id = tankInputTensorEids[0];
                const model = mapTankToModel.get(id)!;
                mapModelToFitness.set(model, (mapModelToFitness.get(model) ?? 0) + 100);
            }

            if (steps-- <= 0 || tankInputTensorEids.length < 2) {
                console.log('>> result', models.map(model => mapModelToFitness.get(model) ?? 0));
                destroy();
                stopInterval();
                resolve(models.map(model => mapModelToFitness.get(model) ?? 0));
                return;
            }

            for (let i = 0; i < tankInputTensorEids.length; i++) {
                const tankEid = tankInputTensorEids[i];

                if (!mapTankToModel.has(tankEid)) {
                    mapTankToModel.set(tankEid, models[i]);
                    mapModelToFitness.set(models[i], 0);
                }

                const model = mapTankToModel.get(tankEid)!;

                const tankX = TankInputTensor.x[tankEid];
                const tankY = TankInputTensor.y[tankEid];
                const inputVector = new Float32Array(65);
                let k = 0;
                // map
                inputVector[k++] = width;
                inputVector[k++] = height;
                // tank
                inputVector[k++] = TankInputTensor.health[tankEid];
                inputVector[k++] = tankX;
                inputVector[k++] = tankY;
                inputVector[k++] = TankInputTensor.speed[tankEid];
                inputVector[k++] = TankInputTensor.rotation[tankEid];
                inputVector[k++] = TankInputTensor.turretRotation[tankEid];
                inputVector[k++] = TankInputTensor.projectileSpeed[tankEid];

                // enemies
                const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
                inputVector.set(enemiesBuffer, k);
                k += enemiesBuffer.length;

                // bullets
                const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
                inputVector.set(bulletsBuffer, k);
                k += bulletsBuffer.length;

                // @ts-ignore
                const inputTensor = tf.tensor2d([inputVector]);
                const outputTensor = model.predict(inputTensor) as tf.Tensor;
                mapTankToOutputTensor.set(tankEid, outputTensor);
                tf.dispose([inputTensor]);
            }

            for (let i = 0; i < tankInputTensorEids.length; i++) {
                const tankEid = tankInputTensorEids[i];
                const tankX = TankInputTensor.x[tankEid];
                const tankY = TankInputTensor.y[tankEid];

                const model = mapTankToModel.get(tankEid)!;
                let fitness = mapModelToFitness.get(model) ?? 0;
                const outputTensor = mapTankToOutputTensor.get(tankEid)!;
                const output = outputTensor.dataSync(); // [shot, move, turn, targetX, targetY]
                tf.dispose([outputTensor]);
                const shouldShot = output[0] > 0;
                TankController.setShooting(tankEid, shouldShot);
                TankController.setMove$(tankEid, output[1]);
                TankController.setRotate$(tankEid, output[2]);
                TankController.setTurretTarget$(
                    tankEid,
                    ((output[3] + 1) / 2) * width,
                    ((output[4] + 1) / 2) * height,
                );

                resetReward();

                // Если цель пушки указывает на один из танков
                const turretTarget = TankController.getTurretTarget(tankEid);
                for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
                    const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 4);
                    const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 4 + 1);

                    // stupid condition
                    if (enemyX !== 0 && enemyY !== 0) {
                        const distFromTargetToEnemy = dist2(turretTarget[0], turretTarget[1], enemyX, enemyY);
                        const distFromTankToEnemy = dist2(tankX, tankY, enemyX, enemyY);
                        reward.aim += max(0, (50 - distFromTargetToEnemy) / 50) * max(0, (1000 - distFromTankToEnemy) / 1000);
                    }
                }

                // reward for avoiding bullets
                for (let j = 0; j < TANK_INPUT_TENSOR_MAX_BULLETS; j++) {
                    const bulletX = TankInputTensor.bulletsData.get(tankEid, j * 4);
                    const bulletY = TankInputTensor.bulletsData.get(tankEid, j * 4 + 1);

                    if (bulletX !== 0 && bulletY !== 0) {
                        const distToBullet = dist2(tankX, tankY, bulletX, bulletY);
                        reward.bulletAvoidance += distToBullet > 10 ? 1 : -1;
                    }
                }

                // reward for staying in the map
                reward.mapStay += inRange(tankX, 0, width) && inRange(tankY, 0, height) ? 2 : -10;

                // если танк живой
                reward.health += TankInputTensor.health[tankEid];

                fitness += reward.aim * 10 + reward.bulletAvoidance * 5 + reward.mapStay * 2 + reward.health;

                mapModelToFitness.set(model, fitness);
            }
        }, TICK_TIME_REAL);
    });
}

// Функция создания модели с заданной архитектурой
function createModel(): Sequential {
    const model = sequential();

    // Входной слой: 29 признаков
    model.add(layers.dense({
        inputShape: [65],
        units: 32,
        activation: 'relu',
        useBias: true,
    }));

    // Скрытый слой
    model.add(layers.dense({
        units: 16,
        activation: 'relu',
        useBias: true,
    }));

    // Выходной слой: 5 выхода с активацией tanh (значения от -1 до 1)
    model.add(layers.dense({
        units: 5,
        activation: 'tanh',
        useBias: true,
    }));

    return model;
}

// Функция клонирования модели (копирует архитектуру и веса)
function cloneModel(model: LayersModel): Sequential {
    const newModel = createModel();
    const weights = model.getWeights();
    const clonedWeights = weights.map(w => w.clone());
    newModel.setWeights(clonedWeights);
    return newModel;
}

// Функция мутации модели: для каждого веса с вероятностью MUTATION_RATE добавляем шум
function mutateModel(model: LayersModel): void {
    const weights = model.getWeights();
    const mutatedWeights = weights.map(t => {
        const vals = t.dataSync();
        const shape = t.shape;
        const newVals = Array.from(vals).map(v => {
            if (Math.random() < MUTATION_RATE) {
                return v + (Math.random() * 0.2 - 0.1);
            }
            return v;
        });
        return tensor(newVals, shape);
    });
    model.setWeights(mutatedWeights);
}

// Функция оценки (fitness) модели через симуляцию игры
async function evaluateModels(models: tf.LayersModel[]): Promise<number[]> {
    // Запустим симуляцию несколько раз, чтобы сгладить результат
    const NUM_SIMULATIONS = 3;
    let totalFitness = new Array(TANK_COUNT_SIMULATION).fill(0);
    for (let i = 0; i < NUM_SIMULATIONS; i++) {
        const fitness = await simulateGame(models);
        for (let j = 0; j < TANK_COUNT_SIMULATION; j++) {
            totalFitness[j] += fitness[j];
        }
    }
    return totalFitness.map(f => f / NUM_SIMULATIONS);
}

// Основная функция эволюционного обучения
async function evolutionaryTraining(): Promise<void> {
    // const prevModel = await tf.loadLayersModel('indexeddb://tank-model');

    // Инициализация популяции
    let population: (tf.LayersModel | Sequential)[] = [];

    for (let i = 0; i < POPULATION_SIZE; i++) {
        // population.push(prevModel ? cloneModel(prevModel) : createModel());
        population.push(createModel());
    }

    // Эволюционный цикл
    for (let gen = 0; gen < GENERATIONS; gen++) {
        // Оценка фитнеса для каждого индивидуума
        const fitnesses: number[] = [];
        for (let i = 0; i < POPULATION_SIZE; i += TANK_COUNT_SIMULATION) {
            fitnesses.push(
                ...(await evaluateModels(population.slice(i, i + TANK_COUNT_SIMULATION))),
            );
        }

        // Вывод статистики поколения
        const bestFitness = Math.max(...fitnesses);
        const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
        console.log(`Поколение ${ gen }: Лучший фитнес = ${ bestFitness.toFixed(3) }, Средний фитнес = ${ avgFitness.toFixed(3) }`);

        // Сортируем индексы по убыванию фитнеса
        const sortedIndices = fitnesses
            .map((fit, idx) => ({ fit, idx }))
            .sort((a, b) => b.fit - a.fit)
            .map(item => item.idx);

        const newPopulation: Sequential[] = [];

        // Элитизм: сохраняем лучших ELITE_COUNT особей без изменений
        for (let i = 0; i < ELITE_COUNT; i++) {
            if (i === 0) {
                console.log(`Лучший фитнес на прогоне ${ gen } сохранен`);
                await population[sortedIndices[i]].save('indexeddb://tank-model');
            }
            newPopulation.push(cloneModel(population[sortedIndices[i]]));
        }


        // Создаем потомство для оставшейся части популяции
        while (newPopulation.length < POPULATION_SIZE) {
            // Выбираем двух родителей из верхней полов популяции
            const parent1 = population[sortedIndices[Math.floor(Math.random() * (POPULATION_SIZE / 2))]];
            const parent2 = population[sortedIndices[Math.floor(Math.random() * (POPULATION_SIZE / 2))]];

            // Создаем ребёнка через кроссовер весов
            const child = createModel();
            const weights1 = parent1.getWeights();
            const weights2 = parent2.getWeights();

            const newWeights = weights1.map((w1, idx) => {
                const w2 = weights2[idx];
                const vals1 = w1.dataSync();
                const vals2 = w2.dataSync();
                const shape = w1.shape;
                const newVals: number[] = [];
                for (let j = 0; j < vals1.length; j++) {
                    newVals.push(Math.random() < 0.5 ? vals1[j] : vals2[j]);
                }
                return tensor(newVals, shape);
            });
            child.setWeights(newWeights);

            // Применяем мутацию
            mutateModel(child);
            newPopulation.push(child);
        }

        // Освобождаем память старой популяции
        population.forEach(model => model.dispose());
        population = newPopulation;
    }


    // // По завершении эволюции выбираем лучшую модель
    const finalFitnesses = await evaluateModels(population);
    const bestIndex = finalFitnesses.indexOf(Math.max(...finalFitnesses));
    console.log('Эволюция завершена. Лучшая модель имеет индекс:', bestIndex);
    const bestModel = population[bestIndex];

    await bestModel.save('indexeddb://tank-model');
    await bestModel.save('downloads://tank-model');
    debugger
}

evolutionaryTraining();
