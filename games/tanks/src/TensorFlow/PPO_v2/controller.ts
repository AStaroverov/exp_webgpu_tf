import { createInputVector } from '../Common/createInputVector';
import { TankController } from '../../ECS/Components/TankController';
import * as tf from '@tensorflow/tfjs';
import { disposeSharedAgent, getSharedAgent } from './agent.ts';
import { calculateMultiHeadReward } from '../Common/calculateMultiHeadReward.ts';

// Map to store previous states for each tank
const tankStates = new Map<number, Float32Array>();
// Map to track previous health values
const prevHealth = new Map<number, number>();
// Map to store previous actions
const prevActions = new Map<number, {
    action: number[];
    logProb: tf.Tensor;
    value: tf.Tensor;
}>();
// Track accumulated rewards per tank
const tankRewards = new Map<number, number>();
// Track active status of tanks
const activeTanks = new Map<number, boolean>();

export function initSharedRLController(_useTrainedModel = true) {
    console.log('Initializing shared PPO controller');

    // Get shared agent
    const agent = getSharedAgent();

    // Clear previous state maps
    tankStates.clear();
    prevHealth.clear();
    prevActions.clear();
    tankRewards.clear();
    activeTanks.clear();

    return agent;
}

export function resetSharedRLController() {
    // Clear previous state maps
    tankStates.clear();
    prevHealth.clear();
    prevActions.clear();
    tankRewards.clear();
    activeTanks.clear();
    // Reset agent
    // const agent = getSharedAgent();
    // agent.dispose();
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
}

/**
 * Update tank using the shared PPO controller
 * @param tankEid Tank entity ID
 * @param width Canvas width
 * @param height Canvas height
 * @param isTraining Whether we're in training mode
 */
export function updateTankWithSharedRL(
    tankEid: number,
    width: number,
    height: number,
    isTraining = false,
) {
    // Check if tank is active
    if (!activeTanks.get(tankEid)) {
        return;
    }

    // Get shared agent
    const agent = getSharedAgent();

    // Create input vector for the current state
    const inputVector = createInputVector(tankEid, width, height);

    // Create tensor from input vector
    const currentState = tf.tensor1d(inputVector);

    try {
        // Get action from agent
        const result = agent.act(currentState, tankEid, isTraining);
        const { action } = result;

        // Apply action to tank controller
        applyActionToTank(tankEid, action, width, height);

        // If in training mode and we have a previous state, calculate reward and store experience
        if (isTraining && tankStates.has(tankEid) && prevActions.has(tankEid)) {
            const prevState = tankStates.get(tankEid)!;
            const prevStateTensor = tf.tensor1d(prevState);
            const { action: prevAction, logProb: prevLogProb, value: prevValue } = prevActions.get(tankEid)!;

            // Calculate reward
            const reward = calculateTotalReward(
                tankEid,
                prevAction,
                prevHealth.get(tankEid)!,
                width,
                height,
            );

            // Accumulate reward for this tank
            const currentReward = tankRewards.get(tankEid) || 0;
            tankRewards.set(tankEid, currentReward + reward);

            // Check if tank is "dead" based on health
            const isDone = inputVector[0] <= 0;

            // Store experience in agent's memory
            agent.remember(
                tankEid,
                prevStateTensor,
                tf.tensor1d(prevAction),
                prevLogProb,
                prevValue,
                reward,
                isDone,
            );
        }

        // Store current action, logProb, and value for next update
        prevActions.set(tankEid, {
            action,
            value: result.value,
            logProb: result.logProb,
        });

        // Update previous health
        prevHealth.set(tankEid, inputVector[0]);

        // Store current state for next update
        tankStates.set(tankEid, new Float32Array(inputVector));

        // Clean up tensors
        currentState.dispose();
    } catch (error) {
        console.error(`Error updating shared PPO controller for tank ${ tankEid }:`, error);
        currentState.dispose();
    }
}

/**
 * Apply the PPO agent's action to the tank controller
 * @param tankEid Tank entity ID
 * @param action Array of action values [shoot, move, rotate, aimX, aimY]
 * @param width Canvas width
 * @param height Canvas height
 */
function applyActionToTank(tankEid: number, action: number[], width: number, height: number) {
    const [shoot, move, rotate, aimDX, aimDY] = action;

    TankController.setShooting$(tankEid, shoot > 0.5);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);

    // Set turret target (aiming)
    const turretTargetX = TankController.turretTarget.get(tankEid, 0) + aimDX * width * 0.05;
    const turretTargetY = TankController.turretTarget.get(tankEid, 1) + aimDY * height * 0.05;
    TankController.setTurretTarget$(tankEid, turretTargetX, turretTargetY);
}

/**
 * Calculates the total reward based on multi-head reward components
 */
function calculateTotalReward(
    tankEid: number,
    action: number[],
    prevHealth: number,
    width: number,
    height: number,
): number {
    // Используем существующую функцию для расчета компонентов наград
    const rewardComponents = calculateMultiHeadReward(
        tankEid,
        action,
        prevHealth,
        width,
        height,
    );

    // Просто возвращаем общую награду
    return rewardComponents.totalReward;
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
    // Free tensors
    if (prevActions.has(tankEid)) {
        const { logProb, value } = prevActions.get(tankEid)!;
        logProb.dispose();
        value.dispose();
    }

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
    // Free tensors
    for (const { logProb, value } of prevActions.values()) {
        logProb.dispose();
        value.dispose();
    }

    // Clear maps
    tankStates.clear();
    prevHealth.clear();
    prevActions.clear();
    tankRewards.clear();
    activeTanks.clear();

    // Dispose shared agent
    disposeSharedAgent();

    console.log('All PPO resources cleaned up');
}

/**
 * Log episode completion
 * @param episodeLength Length of the episode in frames
 * @returns Agent statistics
 */
export function completeAgentEpisode(episodeLength: number) {
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
export async function saveSharedAgent() {
    const agent = getSharedAgent();
    return agent.save();
}