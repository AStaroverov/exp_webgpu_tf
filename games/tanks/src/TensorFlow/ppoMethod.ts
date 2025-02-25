import * as tf from '@tensorflow/tfjs';
import { layers, LayersModel, sequential } from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../ECS/Components/TankState.ts';
import { Tank } from '../ECS/Components/Tank.ts';
import { TankController } from '../ECS/Components/TankController.ts';
import { clamp, inRange } from 'lodash-es';
import { createBattlefield } from './createBattlefield.ts';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';
import { abs, dist2, hypot, max, min } from '../../../../lib/math.ts';
import { query } from 'bitecs';

setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
await tf.setBackend('wasm');

// Configuration constants
export const TANK_COUNT_SIMULATION = 6; // Reduced to make training more manageable
const TICK_TIME_REAL = 1;
const TICK_TIME_SIMULATION = 16.6667 * 2;
const INPUT_DIM = 63; // Tank state dimensions (same as your original implementation)
const ACTION_DIM = 5; // [shoot, move, turn, targetX, targetY]

const PPO_EPOCHS = 4;
const BATCH_SIZE = 64;
const CLIP_EPSILON = 0.2;
const ENTROPY_BETA = 0.01;
const GAMMA = 0.99; // Discount factor
const LAMBDA = 0.95; // GAE parameter

const TANK_RADIUS = 80;

// Experience buffer
class ExperienceBuffer {
    states: tf.Tensor[] = [];
    actions: tf.Tensor[] = [];
    oldLogProbs: tf.Tensor[] = [];
    rewards: number[] = [];
    values: tf.Tensor[] = [];

    constructor(private capacity: number = 1024) {
    }

    get size() {
        return this.states.length;
    }

    add(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, reward: number, value: tf.Tensor) {
        this.states.push(state);
        this.actions.push(action);
        this.oldLogProbs.push(logProb);
        this.rewards.push(reward);
        this.values.push(value);

        // Trim if exceeding capacity
        if (this.states.length > this.capacity) {
            this.states.shift();
            this.actions.shift();
            this.oldLogProbs.shift();
            this.rewards.shift();
            this.values.shift();
        }
    }

    clear() {
        // Clean up tensors
        this.states.forEach(t => t.dispose());
        this.actions.forEach(t => t.dispose());
        this.oldLogProbs.forEach(t => t.dispose());
        this.values.forEach(t => t.dispose());

        this.states = [];
        this.actions = [];
        this.oldLogProbs = [];
        this.rewards = [];
        this.values = [];
    }

    // Calculate advantages using Generalized Advantage Estimation (GAE)
    computeReturnsAndAdvantages(lastValue: number): [tf.Tensor, tf.Tensor] {
        const returns: number[] = new Array(this.rewards.length);
        const advantages: number[] = new Array(this.rewards.length);

        let nextReturn = lastValue;
        let nextAdvantage = 0;

        for (let i = this.rewards.length - 1; i >= 0; i--) {
            const reward = this.rewards[i];
            const value = this.values[i].dataSync()[0];

            // Compute TD error and GAE
            const delta = reward + GAMMA * nextReturn - value;
            nextAdvantage = delta + GAMMA * LAMBDA * nextAdvantage;
            nextReturn = reward + GAMMA * nextReturn;

            returns[i] = nextReturn;
            advantages[i] = nextAdvantage;
        }

        return [
            tf.tensor1d(returns),
            tf.tensor1d(advantages),
        ];
    }

    getBatch(batchSize: number): [tf.Tensor[], tf.Tensor[], tf.Tensor[], tf.Tensor, tf.Tensor] {
        if (this.size < batchSize) {
            throw new Error(`Buffer size (${ this.size }) smaller than requested batch size (${ batchSize })`);
        }

        // Randomly select indices for batch
        const indices = [];
        for (let i = 0; i < batchSize; i++) {
            indices.push(Math.floor(Math.random() * this.size));
        }

        // Get batch elements
        const batchStates = indices.map(i => this.states[i]);
        const batchActions = indices.map(i => this.actions[i]);
        const batchLogProbs = indices.map(i => this.oldLogProbs[i]);

        // Calculate returns and advantages
        const [returns, advantages] = this.computeReturnsAndAdvantages(
            this.values[this.values.length - 1].dataSync()[0],
        );

        const batchReturns = tf.gather(returns, indices);
        const batchAdvantages = tf.gather(advantages, indices);

        // Normalize advantages
        const mean = tf.mean(batchAdvantages);
        const std = tf.sqrt(tf.mean(tf.square(tf.sub(batchAdvantages, mean))));
        const normalizedAdvantages = tf.div(tf.sub(batchAdvantages, mean), tf.add(std, 1e-8));

        returns.dispose();
        advantages.dispose();
        mean.dispose();
        std.dispose();

        return [batchStates, batchActions, batchLogProbs, batchReturns, normalizedAdvantages];
    }
}

// Create Actor (Policy) Network - using separate models for means and std
function createActorModel(): { meanModel: LayersModel, stdModel: LayersModel } {
    // Mean model
    const meanModel = sequential();
    meanModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    meanModel.add(layers.dense({
        units: 64,
        activation: 'relu',
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
        activation: 'tanh',  // Constrain the log std for stability
        biasInitializer: tf.initializers.constant({ value: -0.5 }),  // Initialize with small std
    }));

    // Compile the models individually
    meanModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    stdModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return { meanModel, stdModel };
}

// Create Critic (Value) Network
function createCriticModel(): LayersModel {
    const model = sequential();

    // Input layer
    model.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Hidden layers
    model.add(layers.dense({
        units: 64,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Output layer - single value estimate
    model.add(layers.dense({
        units: 1,
        kernelInitializer: 'glorotNormal',
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
    });

    return model;
}

// Implementation of PPO Agent
class PPOAgent {
    actorMean: LayersModel;
    actorStd: LayersModel;
    critic: LayersModel;
    buffer: ExperienceBuffer;

    constructor() {
        const { meanModel, stdModel } = createActorModel();
        this.actorMean = meanModel;
        this.actorStd = stdModel;
        this.critic = createCriticModel();
        this.buffer = new ExperienceBuffer();
    }

    // Sample action from policy
    sampleAction(state: tf.Tensor): [tf.Tensor, tf.Tensor] {
        return tf.tidy(() => {
            // Get mean and log std from actor networks
            const meanTensor = this.actorMean.predict(state) as tf.Tensor;
            const logStdTensor = this.actorStd.predict(state) as tf.Tensor;

            // Convert to standard deviation
            const stdTensor = tf.exp(logStdTensor);

            // Sample from normal distribution
            const epsilonTensor = tf.randomNormal(meanTensor.shape);
            const actionTensor = tf.add(meanTensor, tf.mul(stdTensor, epsilonTensor));

            // Constrain actions to valid range [-1, 1] if needed
            const clippedActionTensor = tf.clipByValue(actionTensor, -1, 1);

            // Calculate log probability of the action
            const logProbTensor = this.calculateLogProb(meanTensor, stdTensor, clippedActionTensor);

            return [clippedActionTensor, logProbTensor];
        });
    }

    // Calculate log probability of an action
    calculateLogProb(mean: tf.Tensor, std: tf.Tensor, action: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            // Formula: -0.5 * (((action - mean) / std)^2 + 2 * log(std) + log(2*pi))
            const variance = tf.square(std);
            const logVariance = tf.log(variance);

            const diff = tf.sub(action, mean);
            const scaled = tf.div(diff, std);
            const squaredScaled = tf.square(scaled);

            const logProb = tf.mul(
                tf.scalar(-0.5),
                tf.add(
                    tf.add(squaredScaled, tf.mul(tf.scalar(2), logVariance)),
                    tf.scalar(Math.log(2 * Math.PI)),
                ),
            );

            // Sum across action dimensions
            return tf.sum(logProb, -1);
        });
    }

    // Evaluate state with critic
    evaluateState(state: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            return this.critic.predict(state) as tf.Tensor;
        });
    }

    // Store experience in buffer
    storeExperience(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, reward: number, value: tf.Tensor) {
        this.buffer.add(state.clone(), action.clone(), logProb.clone(), reward, value.clone());
    }

    // Train using PPO algorithm
    async train() {
        if (this.buffer.size < BATCH_SIZE) {
            console.log('Not enough samples for training');
            return;
        }

        console.log(`Training PPO with ${ this.buffer.size } samples`);

        for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            // Get training batch
            const [
                batchStates,
                batchActions,
                batchOldLogProbs,
                batchReturns,
                batchAdvantages,
            ] = this.buffer.getBatch(BATCH_SIZE);

            // Optimize actor
            const actorLoss = await this.optimizeActor(
                batchStates,
                batchActions,
                batchOldLogProbs,
                batchAdvantages,
            );

            // Optimize critic
            const criticLoss = await this.optimizeCritic(
                batchStates,
                batchReturns,
            );

            console.log(`Epoch ${ epoch + 1 }/${ PPO_EPOCHS }: Actor Loss = ${ actorLoss.toFixed(4) }, Critic Loss = ${ criticLoss.toFixed(4) }`);

            // Dispose tensors
            batchReturns.dispose();
            batchAdvantages.dispose();
        }

        // Clear buffer after training
        this.buffer.clear();
    }

    // Optimize actor using PPO clip objective
    async optimizeActor(
        states: tf.Tensor[],
        actions: tf.Tensor[],
        oldLogProbs: tf.Tensor[],
        advantages: tf.Tensor,
    ): Promise<number> {
        // Combine states into a batch
        const stateBatch = tf.concat(states.map(s => s.reshape([1, -1])));
        const actionBatch = tf.concat(actions.map(a => a.reshape([1, -1])));
        const oldLogProbBatch = tf.concat(oldLogProbs.map(lp => lp.reshape([1, -1])));

        // Create optimizer
        const meanOptimizer = tf.train.adam(0.0003);
        const stdOptimizer = tf.train.adam(0.0003);

        // Custom training step with gradients
        const gradFunction = () => {
            // TF.js tidy to manage memory
            return tf.tidy(() => {
                // Forward pass to get current policy distribution
                const meanTensor = this.actorMean.predict(stateBatch) as tf.Tensor;
                const logStdTensor = this.actorStd.predict(stateBatch) as tf.Tensor;
                const stdTensor = tf.exp(logStdTensor);

                // Calculate new log probabilities
                const newLogProbs = this.calculateLogProb(meanTensor, stdTensor, actionBatch);

                // Calculate ratio and clipped ratio for PPO
                const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbBatch.reshape([-1])));
                const clippedRatio = tf.clipByValue(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON);

                // PPO loss terms
                const surrogateLoss1 = tf.mul(ratio, advantages);
                const surrogateLoss2 = tf.mul(clippedRatio, advantages);
                const ppoLoss = tf.neg(tf.mean(tf.minimum(surrogateLoss1, surrogateLoss2)));

                // Entropy bonus for exploration
                const entropy = tf.mean(tf.add(logStdTensor, tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E))));
                const totalLoss = tf.sub(ppoLoss, tf.mul(tf.scalar(ENTROPY_BETA), entropy));

                return totalLoss;
            });
        };

        // Train the networks separately using custom optimize function
        const meanVars = this.actorMean.weights.map(w => [w.name, w.read()] as const);
        const stdVars = this.actorStd.weights.map(w => [w.name, w.read()] as const);

        // Apply optimization
        const { value, grads } = tf.variableGrads(
            // @ts-expect-error
            gradFunction,
            [...meanVars.map(([_, t]) => t), ...stdVars.map(([_, t]) => t)],
        );
        // Apply gradients to mean model
        meanOptimizer.applyGradients(
            meanVars.map(([name]) => {
                return { name, tensor: grads[name] };
            }),
        );

        // Apply gradients to std model
        stdOptimizer.applyGradients(
            stdVars.map(([name]) => {
                return { name, tensor: grads[name] };
            }),
        );

        // Cleanup
        stateBatch.dispose();
        actionBatch.dispose();
        oldLogProbBatch.dispose();

        return value.dataSync()[0];
    }

    // Optimize critic using MSE loss
    async optimizeCritic(
        states: tf.Tensor[],
        returns: tf.Tensor,
    ): Promise<number> {
        // Combine states into a batch
        const stateBatch = tf.concat(states.map(s => s.reshape([1, -1])));

        // critic optimizer mutation
        tf.train.adam(0.001);

        // Optimize critic using built-in fit method
        const history = await this.critic.fit(stateBatch, returns, {
            epochs: 1,
            batchSize: states.length,
            verbose: 0,
        });

        // Cleanup
        stateBatch.dispose();

        return history.history.loss[0] as number;
    }

    // Save models
    async saveModels() {
        await this.actorMean.save('indexeddb://tank-actor-mean-model');
        await this.actorStd.save('indexeddb://tank-actor-std-model');
        await this.critic.save('indexeddb://tank-critic-model');
        console.log('Models saved successfully');
    }

    // Load models
    async loadModels() {
        try {
            this.actorMean = await tf.loadLayersModel('indexeddb://tank-actor-mean-model');
            this.actorStd = await tf.loadLayersModel('indexeddb://tank-actor-std-model');
            this.critic = await tf.loadLayersModel('indexeddb://tank-critic-model');
            this.actorMean.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
            this.actorStd.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
            this.critic.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
            console.log('Models loaded successfully');
            return true;
        } catch (error) {
            console.log('Failed to load models:', error);
            return false;
        }
    }
}

// Function to run a simulation episode
async function runEpisode(agent: PPOAgent, maxSteps: number): Promise<number> {
    const episodeReward: number[] = [];
    return new Promise(resolve => {
        const { world, canvas, gameTick, destroy } = createBattlefield(TANK_COUNT_SIMULATION);

        const mapTankToState = new Map<number, tf.Tensor>();
        const mapTankToAction = new Map<number, tf.Tensor>();
        const mapTankToLogProb = new Map<number, tf.Tensor>();
        const mapTankToValue = new Map<number, tf.Tensor>();
        const mapTankToReward = new Map<number, number>();

        let steps = 0;
        const stopInterval = macroTasks.addInterval(() => {
            gameTick(TICK_TIME_SIMULATION);
            steps++;

            // Terminate if episode is too long or all tanks are destroyed
            const tankEids = query(world, [Tank, TankController, TankInputTensor]);
            const onlyOneTankLeft = tankEids.length === 1;

            if (onlyOneTankLeft) {
                // Reward for surviving
                mapTankToReward.set(tankEids[0], mapTankToReward.get(tankEids[0])! * 1.1);
            }

            if (steps >= maxSteps || onlyOneTankLeft) {
                console.log(`Episode ended after ${ steps } steps with avg reward: ${
                    episodeReward.length > 0 ?
                        (episodeReward.reduce((a, b) => a + b, 0) / episodeReward.length).toFixed(2) :
                        'N/A'
                }`);

                destroy();
                stopInterval();

                const totalReward = episodeReward.reduce((a, b) => a + b, 0);
                resolve(totalReward);
                return;
            }

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            const vDelta = 200;
            const vWidth = width + vDelta;
            const vHeight = height + vDelta;
            const maxSpeed = 10_000;

            // Process each tank
            for (let tankEid of tankEids) {
                // Initialize reward for this tank if not existing
                if (!mapTankToReward.has(tankEid)) {
                    mapTankToReward.set(tankEid, 0);
                }

                // Prepare state tensor
                const tankX = TankInputTensor.x[tankEid];
                const tankY = TankInputTensor.y[tankEid];
                const vTankX = tankX + vDelta;
                const vTankY = TankInputTensor.y[tankEid] + vDelta;
                const inputVector = new Float32Array(INPUT_DIM);
                let k = 0;

                // Tank state
                inputVector[k++] = TankInputTensor.health[tankEid];
                inputVector[k++] = clamp(vTankX / vWidth, 0, 1);
                inputVector[k++] = clamp(vTankY / vHeight, 0, 1);
                inputVector[k++] = TankInputTensor.speed[tankEid] / maxSpeed;
                inputVector[k++] = TankInputTensor.rotation[tankEid] / Math.PI;
                inputVector[k++] = TankInputTensor.turretRotation[tankEid] / Math.PI;
                inputVector[k++] = TankInputTensor.projectileSpeed[tankEid] / maxSpeed;

                // Enemies data
                const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
                for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
                    enemiesBuffer[i * 4 + 0] = clamp(enemiesBuffer[i * 4 + 0] / vWidth, 0, 1);
                    enemiesBuffer[i * 4 + 1] = clamp(enemiesBuffer[i * 4 + 1] / vHeight, 0, 1);
                    enemiesBuffer[i * 4 + 2] = enemiesBuffer[i * 4 + 2] / maxSpeed;
                    enemiesBuffer[i * 4 + 3] = enemiesBuffer[i * 4 + 3] / maxSpeed;
                }
                inputVector.set(enemiesBuffer, k);
                k += enemiesBuffer.length;

                // Bullets data
                const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
                for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
                    bulletsBuffer[i * 4 + 0] = clamp(bulletsBuffer[i * 4 + 0] / vWidth, 0, 1);
                    bulletsBuffer[i * 4 + 1] = clamp(bulletsBuffer[i * 4 + 1] / vHeight, 0, 1);
                    bulletsBuffer[i * 4 + 2] = bulletsBuffer[i * 4 + 2] / maxSpeed;
                    bulletsBuffer[i * 4 + 3] = bulletsBuffer[i * 4 + 3] / maxSpeed;
                }
                inputVector.set(bulletsBuffer, k);
                k += bulletsBuffer.length;

                // Create tensor from input vector
                const stateTensor = tf.tensor2d(
                    // @ts-expect-error
                    [inputVector],
                );

                // Store previous state if it exists (for later training)
                const prevState = mapTankToState.get(tankEid);
                const prevAction = mapTankToAction.get(tankEid);
                const prevLogProb = mapTankToLogProb.get(tankEid);
                const prevValue = mapTankToValue.get(tankEid);

                // Get action from policy
                const [actionTensor, logProbTensor] = agent.sampleAction(stateTensor);
                const valueTensor = agent.evaluateState(stateTensor);
                // Apply action to the game
                const actions = actionTensor.dataSync();

                const shouldShoot = actions[0] > 0;
                TankController.setShooting(tankEid, shouldShoot);
                TankController.setMove$(tankEid, actions[1]);
                TankController.setRotate$(tankEid, actions[2]);
                TankController.setTurretTarget$(
                    tankEid,
                    ((actions[3] + 1) / 2) * width,
                    ((actions[4] + 1) / 2) * height,
                );

                // Calculate reward components
                const rewardRecord = {
                    map: 0,
                    aim: 0,
                    avoidBullets: 0,
                    avoidEnemies: 0,
                    health: 0,
                };

                // Reward for staying within map
                if (inRange(tankX, 0, width) && inRange(tankY, 0, height)) {
                    rewardRecord.map += 1;
                }

                // Reward for health
                rewardRecord.health += TankInputTensor.health[tankEid];

                // Reward for aiming at enemies
                const turretTarget = TankController.getTurretTarget(tankEid);
                let hasTargets = false;
                for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
                    const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 4);
                    const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 4 + 1);

                    if (enemyX !== 0 && enemyY !== 0) {
                        hasTargets = true;
                        const distFromTargetToEnemy = dist2(turretTarget[0], turretTarget[1], enemyX, enemyY);
                        const distFromTankToEnemy = dist2(tankX, tankY, enemyX, enemyY);
                        const aimReward =
                            max(0, 1 - distFromTargetToEnemy / TANK_RADIUS)
                            * max(0, 1 - distFromTankToEnemy / 1000);
                        rewardRecord.aim += aimReward;

                        const avoidEnemyReward = 0.2 * min(1, distFromTankToEnemy / (TANK_RADIUS * 3));
                        rewardRecord.avoidEnemies += avoidEnemyReward;
                    }
                }

                if (!hasTargets) {
                    console.warn('No target found');
                }
                if (hasTargets && rewardRecord.aim > 0 && shouldShoot) {
                    rewardRecord.aim += 0.5;
                }

                // Reward for avoiding bullets

                for (let j = 0; j < TANK_INPUT_TENSOR_MAX_BULLETS; j++) {
                    const bX1 = TankInputTensor.bulletsData.get(tankEid, j * 4);
                    const bY1 = TankInputTensor.bulletsData.get(tankEid, j * 4 + 1);
                    const bVX = TankInputTensor.bulletsData.get(tankEid, j * 4 + 2);
                    const bVY = TankInputTensor.bulletsData.get(tankEid, j * 4 + 3);

                    if ((bX1 === 0 && bY1 === 0) || hypot(bVX, bVY) < 100) continue;

                    const distToBullet = hypot(bX1 - tankX, bY1 - tankY);
                    const bX2 = bX1 + bVX;
                    const bY2 = bY1 + bVY;
                    const distToBulletTraverse = abs((bX2 - bX1) * (bY1 - tankX) - (bX1 - tankX) * (bY2 - bY1)) / hypot(bX2 - bX1, bY2 - bY1);

                    if (distToBullet > 300 || distToBulletTraverse > TANK_RADIUS) continue;

                    rewardRecord.avoidBullets += -0.2;
                }

                const reward =
                    rewardRecord.map * 3
                    + rewardRecord.aim * 10
                    + rewardRecord.avoidEnemies * 3
                    + rewardRecord.avoidBullets * 2
                    + rewardRecord.health * 0.1;

                // Store total reward
                const newTotalReward = (mapTankToReward.get(tankEid) || 0) + reward;
                mapTankToReward.set(tankEid, newTotalReward);

                // If we have a previous state-action pair, add it to the buffer
                if (prevState && prevAction && prevLogProb && prevValue) {
                    agent.storeExperience(
                        prevState,
                        prevAction,
                        prevLogProb,
                        reward,
                        prevValue,
                    );

                    // Add to episode reward tracking
                    episodeReward.push(reward);

                    // Cleanup previous tensors
                    prevState.dispose();
                    prevAction.dispose();
                    prevLogProb.dispose();
                    prevValue.dispose();
                }

                // Update state, action map for next step
                mapTankToState.set(tankEid, stateTensor);
                mapTankToAction.set(tankEid, actionTensor);
                mapTankToLogProb.set(tankEid, logProbTensor);
                mapTankToValue.set(tankEid, valueTensor);
            }
        }, TICK_TIME_REAL);
    });

}

// Main PPO training function
async function trainPPO(episodes: number = 100): Promise<void> {
    console.log('Starting PPO training...');

    const agent = new PPOAgent();
    let episodesCompleted = 0;
    let totalReward = 0;

    try {
        // Try to load existing models
        const loaded = await agent.loadModels();
        if (loaded) {
            console.log('Loaded existing models, continuing training');
        } else {
            console.log('Starting with new models');
        }

        // Training loop
        for (let i = 0; i < episodes; i++) {
            console.log(`Starting episode ${ i + 1 }/${ episodes }`);

            // Run episode
            const episodeReward = await runEpisode(agent, 5000);
            totalReward += episodeReward;
            episodesCompleted++;

            console.log(`Episode ${ i + 1 } completed with total reward: ${ episodeReward.toFixed(2) }`);
            console.log(`Average reward so far: ${ (totalReward / episodesCompleted).toFixed(2) }`);

            // Train after each episode
            await agent.train();

            // Save models every 5 episodes
            if ((i + 1) % 5 === 0) {
                await agent.saveModels();
            }
        }

        // Final save
        await agent.saveModels();
        console.log('Training completed!');
    } catch (error) {
        console.error('Error during training:', error);
        setTimeout(() => {
            window.location.reload();
        }, 10_000);
    }
}

// Start the training
trainPPO(1e6);