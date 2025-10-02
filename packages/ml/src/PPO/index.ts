import '@tensorflow/tfjs-backend-wasm';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { createDebugVisualization } from '../../../ml-common/debug.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import '../../../ml-common/uiUtils.ts';
import { restoreModels } from '../Models/Trained/restore.ts';
import { CONFIG } from './config.ts';
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