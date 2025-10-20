import '@tensorflow/tfjs-backend-wasm';
import '../../../../ml-common/unhandledErrors.ts';

import { setConsolePrefix } from '../../../../ml-common/console.ts';
import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';
import { createCritic1Learner, createCritic2Learner } from './createCriticLearner.ts';
import { createLearnerManager } from './createLearnerManager.ts';

import '../../../../ml-common/uiUtils.ts';

/**
 * Worker for training the SAC Critic networks (Q1, Q2).
 * 
 * Responsibilities:
 * - Load critic1, critic2, targetCritic1, targetCritic2, and actor models
 * - Train critics using Bellman backup with entropy regularization
 * - Perform soft target updates: θ_target = τ*θ + (1-τ)*θ_target
 * - Report metrics (critic losses, Q-values, TD errors)
 */

setConsolePrefix(`[LEARNER_CRITIC]`);

await initTensorFlow('webgpu');
createLearnerManager();
// Train both critic networks
createCritic1Learner();
createCritic2Learner();
