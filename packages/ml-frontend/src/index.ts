import '@tensorflow/tfjs-backend-wasm';
import { CONFIG } from '../../ml-common/config.ts';
import { setConsolePrefix } from '../../ml-common/console.ts';
import { createDebugVisualization } from '../../ml-common/debug.ts';
import { initTensorFlow } from '../../ml-common/initTensorFlow.ts';
import '../../ml-common/uiUtils.ts';
import { restoreModels } from '../Models/Trained/restore.ts';
import { VisTestEpisodeManager } from './Actor/VisTest/VisTestEpisodeManager.ts';

setConsolePrefix(`[FRONTEND]`);

await initTensorFlow('wasm');

// TODO: Load models from Supabase instead of local FS
CONFIG.fsModelPath && await restoreModels(CONFIG.fsModelPath);

// Main experience collection: spawn multiple ActorWorkers for fast headless generation
Array.from(
    { length: CONFIG.workerCount },
    () => new Worker(new URL('./src/ActorWorker.ts', import.meta.url), { type: 'module' }),
);

// Visual debug mode: single VisTestEpisodeManager for visualization
const manager = new VisTestEpisodeManager();
manager.start();

createDebugVisualization(document.body, manager);
