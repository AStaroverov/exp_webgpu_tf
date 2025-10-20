import '@tensorflow/tfjs-backend-wasm';
import '../../../../ml-common/unhandledErrors.ts';

import { setConsolePrefix } from '../../../../ml-common/console.ts';
import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';
import { createActorLearner } from './createActorLearner.ts';
import { createLearnerManager } from './createLearnerManager.ts';

import '../../../../ml-common/uiUtils.ts';

/**
 * Worker for training the SAC Actor (policy) network.
 * 
 * Responsibilities:
 * - Load actor, critic1, and critic2 models
 * - Train actor to maximize Q(s, a) - α * log π(a|s)
 * - Use minimum of twin Q-values to prevent overestimation
 * - Report metrics (actor loss, entropy)
 */

setConsolePrefix(`[LEARNER_ACTOR]`);

await initTensorFlow('webgpu');
createLearnerManager();
createActorLearner();
