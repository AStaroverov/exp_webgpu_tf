import '@tensorflow/tfjs-backend-wasm';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';
import { SAC_CONFIG } from '../../../ml-common/config.ts';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { createDebugVisualization } from '../../../ml-common/debug.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import '../../../ml-common/uiUtils.ts';
import { VisTestEpisodeManager } from './VisTest/VisTestEpisodeManager.ts';

setConsolePrefix(`[SAC-TAB]`);

await initTensorFlow('wasm');

// SAC_CONFIG.fsModelPath && await restoreModels(SAC_CONFIG.fsModelPath);

// Create actor workers for data collection
Array.from(
    { length: SAC_CONFIG.workerCount || 4 },
    () => new Worker(new URL('./ActorWorker.ts', import.meta.url), { type: 'module' }),
);

// Delay learner workers startup to allow actors to collect initial data
macroTasks.addTimeout(() => {
    // Actor learner - trains policy network
    new Worker(new URL('./Learner/LearnerActorWorker.ts', import.meta.url), { type: 'module' });

    // Critic learner - trains both Q-networks and updates targets
    new Worker(new URL('./Learner/LearnerCriticWorker.ts', import.meta.url), { type: 'module' });
}, 1000);

// Create visualization/testing manager
const manager = new VisTestEpisodeManager();
manager.start();

createDebugVisualization(document.body, manager);
