import '@tensorflow/tfjs-backend-wasm';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';
import { CONFIG } from '../config.ts';
import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { createDebugVisualization } from '../ui/debug.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import '../ui/uiUtils.ts';
import { VisTestEpisodeManager } from '../agents/VisTestEpisodeManager.ts';

setConsolePrefix(`[TAB]`);

await initTensorFlow('wasm');

// CONFIG.fsModelPath && await restoreModels(CONFIG.fsModelPath);

Array.from(
    { length: CONFIG.workerCount },
    () => new Worker(new URL('./ActorWorker.ts', import.meta.url), { type: 'module' }),
);

macroTasks.addTimeout(() => {
    new Worker(new URL('./LearnerPolicyWorker.ts', import.meta.url), { type: 'module' });
    new Worker(new URL('./LearnerValueWorker.ts', import.meta.url), { type: 'module' });
}, 1000);

const manager = new VisTestEpisodeManager();
manager.start();

createDebugVisualization(document.body, manager);