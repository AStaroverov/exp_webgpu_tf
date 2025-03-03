import { createInputVector } from '../Common/createInputVector';
import { calculateReward } from '../Common/calculateReward_V2';
import { TankController } from '../../ECS/Components/TankController';
import * as tf from '@tensorflow/tfjs';
import { disposeSharedAgent, getSharedAgent } from './agent.ts';

// Map to store previous states for each tank
const tankStates = new Map<number, Float32Array>();
// Map to track previous health values
const prevHealth = new Map<number, number>();
// Map to store previous actions
const prevActions = new Map<number, Float32Array>();
// Track accumulated rewards per tank
const tankRewards = new Map<number, number>();
// Track active status of tanks
const activeTanks = new Map<number, boolean>();

export async function initSharedRLController(_useTrainedModel = true) {
    console.log('Initializing shared RL controller');

    // Get shared agent
    const agent = getSharedAgent();

    // Load trained model if requested
    // if (useTrainedModel) {
    try {
        const loaded = await agent.loadModel();
        if (loaded) {
            console.log('Loaded trained shared model');
        } else {
            console.warn('Failed to load shared model, starting with a new one');
        }
    } catch (error) {
        console.error('Error loading shared model:', error);
    }
    // }

    // Clear previous state maps
    tankStates.clear();
    prevHealth.clear();
    prevActions.clear();
    tankRewards.clear();
    activeTanks.clear();

    return agent;
}

/**
 * Register a tank with the RL controller
 * @param tankEid Tank entity ID
 */
export function registerTank(tankEid: number) {
    // Initialize previous health to full
    prevHealth.set(tankEid, 1.0);

    // Mark tank as active
    activeTanks.set(tankEid, true);

    // Initialize rewards
    tankRewards.set(tankEid, 0);

    console.log(`Tank ${ tankEid } registered with shared RL controller`);
}

/**
 * Update tank using the shared RL controller
 * @param tankEid Tank entity ID
 * @param width Canvas width
 * @param height Canvas height
 * @param maxSpeed Maximum tank speed
 * @param isTraining Whether we're in training mode
 */
export function updateTankWithSharedRL(
    tankEid: number,
    width: number,
    height: number,
    maxSpeed: number,
    isTraining = false,
) {
    // Check if tank is active
    if (!activeTanks.get(tankEid)) {
        return;
    }

    // Get shared agent
    const agent = getSharedAgent();

    // Create input vector for the current state
    const inputVector = createInputVector(tankEid, width, height, maxSpeed);

    // Create tensor from input vector
    const currentState = tf.tensor1d(inputVector);

    try {
        // Get action from agent
        const action = agent.act(currentState, isTraining);
        const actionArray = action.arraySync() as number[];

        // Apply action to tank controller
        applyActionToTank(tankEid, actionArray, width, height);

        // If in training mode and we have a previous state, calculate reward and learn
        if (isTraining && tankStates.has(tankEid)) {
            const prevState = tankStates.get(tankEid)!;
            const prevStateTensor = tf.tensor1d(prevState);

            const prevAction = prevActions.get(tankEid)!;
            const prevActionTensor = tf.tensor1d(prevAction);

            // Calculate reward
            const rewardResult = calculateReward(
                tankEid,
                prevAction,
                prevHealth.get(tankEid)!,
                width,
                height,
            );

            // Accumulate reward for this tank
            const currentReward = tankRewards.get(tankEid) || 0;
            tankRewards.set(tankEid, currentReward + rewardResult.totalReward);

            // Store experience in agent's memory
            const isDone = inputVector[0] <= 0; // Check if tank is "dead" based on health
            agent.remember(
                prevStateTensor,
                prevActionTensor,
                rewardResult.totalReward,
                currentState,
                isDone,
            );

            // Clean up tensors
            prevStateTensor.dispose();
            prevActionTensor.dispose();
        }

        // Store current action for next update
        prevActions.set(tankEid, new Float32Array(actionArray));

        // Update previous health
        prevHealth.set(tankEid, inputVector[0]);

        // Store current state for next update
        tankStates.set(tankEid, new Float32Array(inputVector));

        // Clean up tensors
        currentState.dispose();
        action.dispose();
    } catch (error) {
        console.error(`Error updating shared RL controller for tank ${ tankEid }:`, error);
        currentState.dispose();
    }
}

/**
 * Apply the RL agent's action to the tank controller
 * @param tankEid Tank entity ID
 * @param action Array of action values [shoot, moveX, moveY, aimX, aimY]
 * @param width Canvas width
 * @param height Canvas height
 */
function applyActionToTank(tankEid: number, action: number[], width: number, height: number) {
    const [shoot, moveX, moveY, aimX, aimY] = action;
    TankController.setShooting$(tankEid, shoot > 0.5);

    // Convert normalized coordinates to screen coordinates
    const targetX = (moveX + 1) / 2 * width;
    const targetY = (moveY + 1) / 2 * height;

    // Set movement target
    TankController.setMoveTarget$(tankEid, targetX, targetY);

    // Set turret target (aiming)
    const turretTargetX = (aimX + 1) / 2 * width;
    const turretTargetY = (aimY + 1) / 2 * height;
    TankController.setTurretTarget$(tankEid, turretTargetX, turretTargetY);
}

export function isActiveTank(tankEid: number): boolean {
    return activeTanks.get(tankEid) || false;
}

/**
 * Deactivate a tank (e.g., when destroyed)
 * @param tankEid Tank entity ID
 */
export function deactivateTank(tankEid: number) {
    // Mark tank as inactive
    activeTanks.set(tankEid, false);
    console.log(`Tank ${ tankEid } deactivated`);
}

/**
 * Get number of active tanks
 */
export function getActiveTankCount(): number {
    let count = 0;
    for (const active of activeTanks.values()) {
        if (active) count++;
    }
    return count;
}

/**
 * Train the shared model on the collected experiences
 */
export async function trainSharedModel() {
    const agent = getSharedAgent();
    return agent.train();
}

/**
 * Clean up resources for a specific tank
 * @param tankEid Tank entity ID
 */
export function cleanupTankRL(tankEid: number) {
    // Remove from maps
    tankStates.delete(tankEid);
    prevHealth.delete(tankEid);
    prevActions.delete(tankEid);
    tankRewards.delete(tankEid);
    activeTanks.delete(tankEid);
}

/**
 * Clean up all RL resources
 */
export function cleanupAllRL() {
    // Clear maps
    tankStates.clear();
    prevHealth.clear();
    prevActions.clear();
    tankRewards.clear();
    activeTanks.clear();

    // Dispose shared agent
    disposeSharedAgent();

    console.log('All RL resources cleaned up');
}

/**
 * Log episode completion
 * @param episodeLength Length of the episode in frames
 * @returns Agent statistics
 */
export function logEpisodeCompletion(episodeLength: number) {
    // Calculate average reward across all tanks
    let totalReward = 0;
    let tankCount = 0;

    for (const reward of tankRewards.values()) {
        totalReward += reward;
        tankCount++;
    }

    const avgReward = tankCount > 0 ? totalReward / tankCount : 0;

    // Get shared agent
    const agent = getSharedAgent();

    // Log episode and update epsilon
    agent.logEpisode(avgReward, episodeLength);
    agent.updateEpsilon();

    // Return current stats
    return agent.getStats();
}

/**
 * Save the shared model
 */
export async function saveSharedModel() {
    const agent = getSharedAgent();
    return agent.saveModel();
}