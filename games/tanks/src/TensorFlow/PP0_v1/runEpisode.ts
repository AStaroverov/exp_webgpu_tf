// Function to run a simulation episode with improved handling and rewards
import { PPOAgent } from './PPOAgent.ts';
import { createBattlefield } from './createBattlefield.ts';
import { INPUT_DIM, SKIP_TICKS, TANK_COUNT_SIMULATION, TICK_TIME_REAL, TICK_TIME_SIMULATION } from './consts.ts';
import * as tf from '@tensorflow/tfjs';
import { query } from 'bitecs';
import { Tank } from '../../ECS/Components/Tank.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { calculateReward } from './rewards.ts';

const MAX_SPEED = 10_000;

export async function runEpisode(agent: PPOAgent, maxSteps: number): Promise<number> {
    const { world, canvas, gameTick, destroy } = createBattlefield(TANK_COUNT_SIMULATION);

    const episodeReward: number[] = [];
    const episodeRewardComponents: Record<string, number[]> = {
        map: [],
        aim: [],
        avoidBullets: [],
        avoidEnemies: [],
        health: [],
        damageDealt: [],
        movement: [],
        survival: [],
    };
    // Maps to track tank state and metrics
    const mapTankToState = new Map<number, tf.Tensor>();
    const mapTankToAction = new Map<number, tf.Tensor>();
    const mapTankToLogProb = new Map<number, tf.Tensor>();
    const mapTankToValue = new Map<number, tf.Tensor>();
    const mapTankToReward = new Map<number, number>();
    const mapTankToHealth = new Map<number, number>(); // Track previous health for damage calculation
    const mapTankToDone = new Map<number, boolean>(); // Track terminal states

    const destroyEpisode = () => {
        destroy();
    };

    return new Promise((resolve, reject) => {

        // initial game tick
        gameTick(TICK_TIME_SIMULATION);

        let steps = 0;
        let tankEids = query(world, [Tank, TankController, TankInputTensor]);

        // Initialize health tracking for all tanks
        for (let tankEid of tankEids) {
            mapTankToHealth.set(tankEid, TankInputTensor.health[tankEid]);
            mapTankToDone.set(tankEid, false);
        }

        const stopInterval = macroTasks.addInterval(() => {
            try {
                steps++;

                // Terminate if episode is too long or all tanks are destroyed
                const onlyOneTankLeft = tankEids.length === 1;
                if (steps >= maxSteps || onlyOneTankLeft) {
                    if (onlyOneTankLeft) {
                        // Reward for surviving
                        const survivorEid = tankEids[0];
                        const baseReward = mapTankToReward.get(survivorEid) || 0;
                        const survivalBonus = 5.0; // Существенный бонус за победу
                        mapTankToReward.set(survivorEid, baseReward + survivalBonus);

                        // Логируем победителя
                        console.log(`Tank ${ survivorEid } survived and got a bonus: ${ survivalBonus }`);
                    }

                    // Статистика эпизода
                    const rewardStats = Object.entries(episodeRewardComponents).map(([key, values]) => {
                        const sum = values.reduce((a, b) => a + b, 0);
                        const avg = values.length > 0 ? sum / values.length : 0;
                        return `${ key }: ${ avg.toFixed(2) }`;
                    }).join(', ');

                    console.log(`Episode ended after ${ steps } steps with avg reward: ${
                        episodeReward.length > 0 ?
                            (episodeReward.reduce((a, b) => a + b, 0) / episodeReward.length).toFixed(2) :
                            'N/A'
                    }`);
                    console.log(`Reward components: ${ rewardStats }`);

                    destroyEpisode();
                    stopInterval();

                    const totalReward = episodeReward.reduce((a, b) => a + b, 0);
                    resolve(totalReward);
                    return;
                }

                const width = canvas.offsetWidth;
                const height = canvas.offsetHeight;

                // PHASE 1: Process each tank for state evaluation and action determination
                for (let tankEid of tankEids) {
                    // Initialize reward for this tank if not existing
                    if (!mapTankToReward.has(tankEid)) {
                        mapTankToReward.set(tankEid, 0);
                    }

                    // Check if tank is in a terminal state (flag from previous tick)
                    if (mapTankToDone.get(tankEid)) continue;

                    const inputVector = createInputVector(tankEid, width, height, MAX_SPEED);

                    // Normalize input using the agent's input normalizer
                    const normalizedInputVector = agent.inputNormalizer.normalize(inputVector);

                    // Update normalizer with new data
                    agent.inputNormalizer.update(inputVector);

                    // Create tensor from normalized input vector
                    const stateTensor = tf.tensor2d(
                        // @ts-ignore
                        [normalizedInputVector],
                    );

                    // Get action from policy
                    const [actionTensor, logProbTensor] = agent.sampleAction(stateTensor);
                    const valueTensor = agent.evaluateState(stateTensor);
                    const actions = actionTensor.dataSync();

                    applyActions(tankEid, actions, width, height);

                    // Store tensors for later use after the game tick
                    mapTankToState.set(tankEid, stateTensor);
                    mapTankToAction.set(tankEid, actionTensor);
                    mapTankToLogProb.set(tankEid, logProbTensor);
                    mapTankToValue.set(tankEid, valueTensor);
                }

                // PHASE 2: Execute game tick after all controller updates
                gameTick(TICK_TIME_SIMULATION);
                for (let i = 0; i < SKIP_TICKS; i++) {
                    gameTick(TICK_TIME_SIMULATION);
                }

                // Get list of tanks after the game tick (some might be destroyed)
                const previousTankEids = tankEids;
                tankEids = [...query(world, [Tank, TankController, TankInputTensor])];

                // Find tanks that were destroyed in this tick
                const destroyedTanks = previousTankEids.filter(eid => !tankEids.includes(eid));

                // Add terminal states for destroyed tanks
                for (let destroyedEid of destroyedTanks) {
                    mapTankToDone.set(destroyedEid, true);

                    // Apply death penalty to destroyed tanks
                    const prevReward = mapTankToReward.get(destroyedEid) || 0;
                    const prevState = mapTankToState.get(destroyedEid);
                    const prevAction = mapTankToAction.get(destroyedEid);
                    const prevLogProb = mapTankToLogProb.get(destroyedEid);
                    const prevValue = mapTankToValue.get(destroyedEid);

                    const totalReward = prevReward - 3.0;
                    mapTankToReward.set(destroyedEid, totalReward);

                    // If we have a valid state-action pair from previous step, add it to the buffer
                    if (prevState && prevAction && prevLogProb && prevValue) {
                        agent.storeExperience(
                            prevState,
                            prevAction,
                            prevLogProb,
                            totalReward,
                            prevValue,
                            true,
                        );

                        // Add to episode reward tracking
                        episodeReward.push(totalReward);
                    }

                    console.log(`Tank ${ destroyedEid } was destroyed, terminal state applied`);
                }

                // PHASE 3: Calculate rewards and store experiences after the game tick
                for (let tankEid of tankEids) {
                    // Get previous health
                    const prevHealth = mapTankToHealth.get(tankEid) || 1.0;
                    const currentHealth = TankInputTensor.health[tankEid];

                    // Update health tracking
                    mapTankToHealth.set(tankEid, currentHealth);

                    // Get previous action for reward calculation
                    const actions = mapTankToAction.get(tankEid)?.dataSync();

                    // Calculate reward using the enhanced reward function
                    const { totalReward, rewards } = calculateReward(
                        tankEid,
                        actions || [0, 0, 0, 0, 0],
                        prevHealth,
                        width,
                        height,
                    );

                    // Keep track of individual reward components for debugging
                    Object.entries(rewards).forEach(([key, value]) => {
                        if (episodeRewardComponents[key]) {
                            episodeRewardComponents[key].push(value);
                        }
                    });

                    // Add to total reward for this tank
                    const newTotalReward = (mapTankToReward.get(tankEid) || 0) + totalReward;
                    mapTankToReward.set(tankEid, newTotalReward);

                    // Get previous tensor states
                    const prevState = mapTankToState.get(tankEid);
                    const prevAction = mapTankToAction.get(tankEid);
                    const prevLogProb = mapTankToLogProb.get(tankEid);
                    const prevValue = mapTankToValue.get(tankEid);

                    // If we have a valid state-action pair from previous step, add it to the buffer
                    if (prevState && prevAction && prevLogProb && prevValue) {
                        agent.storeExperience(
                            prevState,
                            prevAction,
                            prevLogProb,
                            totalReward,
                            prevValue,
                            false,
                        );

                        // Add to episode reward tracking
                        episodeReward.push(totalReward);
                    }
                }
            } catch (error) {
                console.error('Error in game tick:', error);
                destroyEpisode();
                stopInterval();
                reject(error);
            }
        }, TICK_TIME_REAL);
    });
}

function createInputVector(tankEid: number, width: number, height: number, maxSpeed: number) {
    const inputVector = new Float32Array(INPUT_DIM);
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    let k = 0;

    // Tank state
    inputVector[k++] = TankInputTensor.health[tankEid];
    inputVector[k++] = tankX / width;
    inputVector[k++] = tankY / height;
    inputVector[k++] = TankInputTensor.speed[tankEid] / maxSpeed;
    inputVector[k++] = TankInputTensor.rotation[tankEid] / Math.PI;
    inputVector[k++] = TankInputTensor.turretRotation[tankEid] / Math.PI;
    inputVector[k++] = TankInputTensor.projectileSpeed[tankEid] / maxSpeed;

    // Enemies data
    const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        enemiesBuffer[i * 4 + 0] = enemiesBuffer[i * 4 + 0] / width;
        enemiesBuffer[i * 4 + 1] = enemiesBuffer[i * 4 + 1] / height;
        enemiesBuffer[i * 4 + 2] = enemiesBuffer[i * 4 + 2] / maxSpeed;
        enemiesBuffer[i * 4 + 3] = enemiesBuffer[i * 4 + 3] / maxSpeed;
    }
    inputVector.set(enemiesBuffer, k);
    k += enemiesBuffer.length;

    // Bullets data
    const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        bulletsBuffer[i * 4 + 0] = bulletsBuffer[i * 4 + 0] / width;
        bulletsBuffer[i * 4 + 1] = bulletsBuffer[i * 4 + 1] / height;
        bulletsBuffer[i * 4 + 2] = bulletsBuffer[i * 4 + 2] / maxSpeed;
        bulletsBuffer[i * 4 + 3] = bulletsBuffer[i * 4 + 3] / maxSpeed;
    }
    inputVector.set(bulletsBuffer, k);
    k += bulletsBuffer.length;

    return inputVector;
}

function applyActions(tankEid: number, actions: ArrayLike<number>, width: number, height: number) {
    const shouldShoot = actions[0] > 0;
    TankController.setShooting(tankEid, shouldShoot);
    TankController.setMove$(tankEid, actions[1]);
    TankController.setRotate$(tankEid, actions[2]);
    TankController.setTurretTarget$(
        tankEid,
        ((actions[3] + 1) / 2) * width,
        ((actions[4] + 1) / 2) * height,
    );
}