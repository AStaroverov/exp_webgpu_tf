import { createInputVector } from '../Common/createInputVector';
import { TankController } from '../../ECS/Components/TankController';
import * as tf from '@tensorflow/tfjs';
import { disposeSharedAgent, getSharedAgent } from './agent.ts';
import { getTankHealth } from '../../ECS/Components/Tank.ts';
import { calculateReward } from '../Common/calculateReward.ts';

// Map to store previous actions
const mapLastUpdateData = new Map<number, {
    action: number[];
}>();
// Track accumulated rewards per tank
const tankRewards = new Map<number, number>();
// Track active status of tanks
const activeTanks = new Map<number, boolean>();

export function initController(_useTrainedModel = true) {
    console.log('Initializing shared PPO controller');

    // Get shared agent
    const agent = getSharedAgent();

    // Clear previous state maps
    mapLastUpdateData.clear();
    tankRewards.clear();
    activeTanks.clear();

    return agent;
}

export function resetController() {
    // Clear previous state maps
    mapLastUpdateData.clear();
    tankRewards.clear();
    activeTanks.clear();
}

/**
 * Register a tank with the RL controller
 * @param tankEid Tank entity ID
 */
export function registerTank(tankEid: number) {
    // Mark tank as active
    activeTanks.set(tankEid, true);

    // Initialize rewards
    tankRewards.set(tankEid, 0);
}

export function updateTankBehaviour(
    tankEid: number,
    width: number,
    height: number,
    isWarmup: boolean,
) {
    // Get shared agent
    const agent = getSharedAgent();
    // Create input vector for the current state
    const inputVector = createInputVector(tankEid, width, height);
    // Get action from agent
    const result = agent.act(inputVector);
    // Apply action to tank controller
    applyActionToTank(tankEid, result.action, width, height);
    mapLastUpdateData.set(tankEid, { action: result.action });

    !isWarmup && agent.rememberAction(
        tankEid,
        tf.tensor1d(inputVector),
        tf.tensor1d(result.action),
        result.logProb,
        result.value,
    );
}

export function memorizeTankBehaviour(
    tankEid: number,
    width: number,
    height: number,
    episode: number,
    rewardMultiplier: number,
    isLast: boolean,
) {
    // Get shared agent
    const agent = getSharedAgent();
    const { action } = mapLastUpdateData.get(tankEid)!;

    // Calculate reward
    const reward = calculateReward(
        tankEid,
        action,
        width,
        height,
        episode,
    ).totalReward;
    // Check if tank is "dead" based on health
    const isDone = getTankHealth(tankEid) <= 0;

    // Accumulate reward for this tank
    tankRewards.set(tankEid, (tankRewards.get(tankEid) || 0) + reward);

    // Store experience in agent's memory
    agent.rememberReward(
        tankEid,
        reward * rewardMultiplier,
        isDone,
        isLast,
    );
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
    const turretTargetX = TankController.turretTarget.get(tankEid, 0) + aimDX * width * 0.01;
    const turretTargetY = TankController.turretTarget.get(tankEid, 1) + aimDY * height * 0.01;
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
export async function trainSharedModel(episode: number) {
    const agent = getSharedAgent();
    return agent.train(episode);
}

/**
 * Clean up all RL resources
 */
export function cleanupAllRL() {
    // Clear maps
    mapLastUpdateData.clear();
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