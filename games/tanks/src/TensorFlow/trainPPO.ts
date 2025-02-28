import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { MAX_STEPS } from './consts.ts';
import { BATCH_SIZE, PPOAgent } from './PPOAgent.ts';
import { runEpisode } from './runEpisode.ts';
import { createActorModel, createExplorationBiasedActorModel, resetPartialWeights } from './models.ts';
import { RingBuffer } from 'ring-buffer-ts';

setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
await tf.setBackend('wasm');

// Конфигурация обучения
const TRAINING_CONFIG = {
    CHECKPOINT_INTERVAL: 5,         // Сохранять каждые N эпизодов
    EXPLORATION_RESET_INTERVAL: 500, // Периодически усиливать исследование
    STALL_DETECTION_WINDOW: 20,     // Окно для обнаружения застоя
    STALL_THRESHOLD: 0.05,          // Порог улучшения для определения застоя
    MAX_TRAIN_ITERATIONS: 5,        // Максимальное количество итераций обучения за эпизод
    MIN_BATCHES_FOR_TRAINING: 3,    // Минимальное количество батчей для начала обучения
    MODEL_VERSION_INTERVAL: 100,    // Интервал для сохранения версионных моделей
    WARMUP_EPISODES: 10,            // Количество эпизодов разминки перед обучением
    RECOVERY_WAIT_TIME: 5000,       // Время ожидания после ошибки (мс)
    MAJOR_VERSION_INTERVAL: 200,   // Интервал для сохранения основных версий модели
};

// Определение типа для состояния обучения (для восстановления)
interface TrainingState {
    currentEpisode: number;
    bestReward: number;
    lastCheckpoint: number;
    explorationModeActive: boolean;
    explorationEndEpisode: number | null;
}

// Полная обновленная функция обучения с улучшенными механизмами
async function trainPPO(): Promise<void> {
    console.log('Запуск улучшенного PPO обучения...');

    // Попытка загрузки предыдущего состояния обучения
    let trainingState: TrainingState = {
        currentEpisode: 0,
        bestReward: -Infinity,
        lastCheckpoint: 0,
        explorationModeActive: false,
        explorationEndEpisode: null,
    };

    try {
        const savedState = localStorage.getItem('tank-training-state');
        if (savedState) {
            trainingState = JSON.parse(savedState);
            console.log(`Загружено предыдущее состояние обучения: эпизод ${ trainingState.currentEpisode }`);
        }
    } catch (error) {
        console.warn('Не удалось загрузить предыдущее состояние обучения:', error);
    }

    // Создаем агента
    const agent = new PPOAgent(true);
    const episodeRewards = new RingBuffer<number>(100);

    try {
        // Пытаемся загрузить существующие модели
        const loaded = await agent.loadModels();
        if (!loaded) {
            console.log('Начинаем с новыми моделями');

            // Если начинаем с нуля, активируем режим исследования на первые 100 эпизодов
            trainingState.explorationModeActive = true;
            trainingState.explorationEndEpisode = trainingState.currentEpisode + 100;

            // Используем модель с усиленным исследованием
            const { meanModel, stdModel } = createExplorationBiasedActorModel();
            agent.actorMean = meanModel;
            agent.actorStd = stdModel;
            console.log('Активирован режим усиленного исследования для начала обучения');
        }

        // Основной цикл обучения
        while (true) {
            trainingState.currentEpisode++;

            // Проверяем, нужно ли выйти из режима исследования
            if (
                trainingState.explorationModeActive
                && trainingState.explorationEndEpisode !== null
                && trainingState.currentEpisode >= trainingState.explorationEndEpisode
            ) {

                console.log('Выход из режима усиленного исследования');
                trainingState.explorationModeActive = false;

                // Если мы не начинали с нуля, восстановим сохраненную модель
                if (trainingState.currentEpisode > 100) {
                    await agent.loadModels('before_exploration');
                    console.log('Восстановлена модель перед режимом исследования');
                } else {
                    // Если это начальное обучение, просто уменьшаем исследование
                    const { meanModel, stdModel } = createActorModel();
                    // Сохраняем веса mean модели, чтобы не потерять прогресс
                    const meanWeights = agent.actorMean.getWeights();
                    agent.actorMean = meanModel;
                    agent.actorMean.setWeights(meanWeights);
                    agent.actorStd = stdModel;
                }
            }

            console.log(`Запуск эпизода ${ trainingState.currentEpisode }`);

            try {
                // Запускаем эпизод
                const episodeReward = await runEpisode(agent, MAX_STEPS);

                episodeRewards.add(episodeReward);

                // После добавления награды эпизода в массив
                if (episodeRewards.isFull()) {
                    const avgReward = episodeRewards.toArray().reduce((a, b) => a + b, 0) / episodeRewards.getPos();
                    console.log(`Средняя награда (100 эпизодов): ${ avgReward.toFixed(4) }`);
                }

                // И проверку на застой:
                if (
                    trainingState.currentEpisode > 200
                    && trainingState.currentEpisode % 20 === 0
                    && isTrainingStalled(episodeRewards, TRAINING_CONFIG.STALL_DETECTION_WINDOW)
                ) {
                    console.log('Обнаружен застой в обучении. Применяем корректирующие меры...');

                    // Здесь добавьте стратегию восстановления, например:
                    await resetPartialWeights(agent.actorMean, 0.3);
                    await resetPartialWeights(agent.critic, 0.3);
                }

                // Обучение на собранном опыте
                if (trainingState.currentEpisode >= TRAINING_CONFIG.WARMUP_EPISODES) {
                    const trainingStartTime = performance.now();
                    let actorLossSum = 0;
                    let criticLossSum = 0;
                    let trainingIterations = 0;

                    // Определяем, достаточно ли данных для обучения
                    const minSamplesRequired = BATCH_SIZE * TRAINING_CONFIG.MIN_BATCHES_FOR_TRAINING;

                    if (agent.buffer.size >= minSamplesRequired) {
                        // Вычисляем оптимальное количество итераций обучения
                        const trainIterations = Math.min(
                            Math.floor(agent.buffer.size / BATCH_SIZE),
                            TRAINING_CONFIG.MAX_TRAIN_ITERATIONS,
                        );

                        for (let iter = 0; iter < trainIterations; iter++) {
                            // Обучаем с текущей скоростью обучения
                            const {
                                actorLoss,
                                criticLoss,
                            } = await agent.train(getLearningRate(trainingState.currentEpisode));

                            if (actorLoss !== undefined && criticLoss !== undefined) {
                                actorLossSum += actorLoss;
                                criticLossSum += criticLoss;
                                trainingIterations++;
                            }
                        }

                        // Логируем средние потери
                        if (trainingIterations > 0) {
                            const avgActorLoss = actorLossSum / trainingIterations;
                            const avgCriticLoss = criticLossSum / trainingIterations;

                            console.log(`Обучение завершено (${ trainingIterations } итераций, ${ (performance.now() - trainingStartTime).toFixed(0) }мс)`);
                            console.log(`Средние потери: Actor = ${ avgActorLoss.toFixed(4) }, Critic = ${ avgCriticLoss.toFixed(4) }`);
                        } else {
                            console.log('В этом эпизоде не было валидного обучения');
                        }
                    } else {
                        console.log(`Недостаточно примеров для обучения: ${ agent.buffer.size }/${ minSamplesRequired }`);
                    }
                }

                // Проверяем, лучший ли это эпизод
                if (episodeReward > trainingState.bestReward) {
                    trainingState.bestReward = episodeReward;

                    // Сохраняем лучшую модель
                    await agent.saveModels('best');
                    console.log(`Новая лучшая модель сохранена с наградой: ${ trainingState.bestReward.toFixed(2) }`);
                }

                // Сохраняем состояние обучения
                localStorage.setItem('tank-training-state', JSON.stringify(trainingState));

                // Регулярное создание контрольных точек
                if ((trainingState.currentEpisode) % TRAINING_CONFIG.CHECKPOINT_INTERVAL === 0) {
                    await agent.saveModels();
                    console.log(`Контрольная точка сохранена на эпизоде ${ trainingState.currentEpisode }`);
                    trainingState.lastCheckpoint = trainingState.currentEpisode;
                }

                // Сохраняем версионную модель с номером эпизода
                if ((trainingState.currentEpisode) % TRAINING_CONFIG.MODEL_VERSION_INTERVAL === 0) {
                    await agent.saveModels(`ep${ trainingState.currentEpisode }`);
                    console.log(`Версионная модель ep${ trainingState.currentEpisode } сохранена`);
                }

                // Периодический сброс для усиления исследования
                if (
                    trainingState.currentEpisode % TRAINING_CONFIG.EXPLORATION_RESET_INTERVAL === 0
                    && !trainingState.explorationModeActive
                ) {

                    console.log(`Плановое усиление исследования на эпизоде ${ trainingState.currentEpisode }`);

                    // Сохраняем текущую модель
                    await agent.saveModels(`before_exploration_${ trainingState.currentEpisode }`);

                    // Применяем частичный сброс весов к stdModel для увеличения исследования
                    await resetPartialWeights(agent.actorStd, 0.5);

                    // Добавляем положительное смещение к выходному слою стандартных отклонений
                    const lastLayer = agent.actorStd.layers[agent.actorStd.layers.length - 1] as tf.layers.Layer;
                    const biasWeights = lastLayer.getWeights()[1];
                    const newBiasWeights = tf.tidy(() => {
                        return tf.add(biasWeights, tf.ones(biasWeights.shape).mul(tf.scalar(0.5)));
                    });

                    lastLayer.setWeights([
                        lastLayer.getWeights()[0],
                        newBiasWeights,
                    ]);

                    console.log('Исследование временно усилено');
                    newBiasWeights.dispose();
                }

                // Для основных версий моделей (каждые 1000 эпизодов)
                if (trainingState.currentEpisode % TRAINING_CONFIG.MAJOR_VERSION_INTERVAL === 0) {
                    const majorVersion = Math.floor(trainingState.currentEpisode / TRAINING_CONFIG.MAJOR_VERSION_INTERVAL);
                    await agent.saveModels(`v${ majorVersion }`);
                    console.log(`Основная версия модели v${ majorVersion } сохранена`);

                    // Перезагрузка страницы для борьбы с утечкой памяти в TensorFlow.js
                    console.log('Перезагрузка страницы для очистки памяти...');
                    window.location.reload();
                    return; // Выход из функции, т.к. страница будет перезагружена
                }
            } catch (error) {
                console.error(`Ошибка в эпизоде ${ trainingState.currentEpisode }:`, error);

                // Пытаемся сохранить текущее состояние перед восстановлением
                try {
                    await agent.saveModels('recovery');
                    console.log('Сохранена точка восстановления');
                } catch (saveError) {
                    console.error('Не удалось сохранить точку восстановления:', saveError);
                }

                // Сохраняем состояние обучения
                localStorage.setItem('tank-training-state', JSON.stringify(trainingState));

                // Ждем перед продолжением
                await new Promise(resolve => setTimeout(resolve, TRAINING_CONFIG.RECOVERY_WAIT_TIME));
            }
        }
    } catch (error) {
        console.error('Критическая ошибка во время обучения:', error);

        // Пытаемся экстренно сохранить
        try {
            await agent.saveModels('emergency');
            console.log('Экстренное сохранение выполнено');
        } catch (saveError) {
            console.error('Не удалось выполнить экстренное сохранение:', saveError);
        }

        // Сохраняем состояние обучения
        localStorage.setItem('tank-training-state', JSON.stringify(trainingState));

        // Ждем перед перезагрузкой страницы
        window.location.reload();
    } finally {
        // Заканчиваем область отслеживания тензоров
        tf.engine().endScope();
    }
}

function getLearningRate(episodeNum: number): number {
    // Начинаем с 0.0003 и медленно снижаем до 0.00006 (1/5 от начальной)
    const initialLR = 0.0003;
    const minLR = 0.00006;
    const decay = 3000; // Более медленный спад - за 3000 эпизодов

    return Math.max(minLR, initialLR * Math.exp(-episodeNum / decay));
}

function isTrainingStalled(rewards: RingBuffer<number>, window: number): boolean {
    if (rewards.getBufferLength() < window * 2) return false;

    const doubleWindowRewards = rewards.getLastN(window * 2);
    const recentRewards = doubleWindowRewards.slice(-window);
    const prevRewards = doubleWindowRewards.slice(0, window);

    const recentAvg = recentRewards.reduce((a, b) => a + b, 0) / window;
    const prevAvg = prevRewards.reduce((a, b) => a + b, 0) / window;

    return recentAvg <= prevAvg || (recentAvg - prevAvg) / Math.abs(prevAvg) < TRAINING_CONFIG.STALL_THRESHOLD;
}

// Запуск обучения
trainPPO();

