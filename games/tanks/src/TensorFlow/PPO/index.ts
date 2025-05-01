import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { createDebugVisualization } from '../Common/debug.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { CONFIG } from './config.ts';
import { restoreModels } from '../Models/Trained/restore.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { VisTestEpisodeManager } from './VisTest/VisTestEpisodeManager.ts';

setConsolePrefix(`[TAB]`);

await initTensorFlow('wasm');

CONFIG.fsModelPath && await restoreModels(CONFIG.fsModelPath);

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