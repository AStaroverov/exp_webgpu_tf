import '@tensorflow/tfjs-backend-wasm';
import '../../ml-common/uiUtils.ts';

import { setConsolePrefix } from '../../ml-common/console.ts';
import { createDebugVisualization } from '../../ml-common/debug.ts';
import { initTensorFlow } from '../../ml-common/initTensorFlow.ts';
import { VisTestEpisodeManager } from './Actor/VisTestEpisodeManager.ts';

setConsolePrefix(`[TAB]`);

await initTensorFlow('wasm');

// Main experience collection: spawn multiple ActorWorkers for fast headless generation
// Array.from(
//     { length: CONFIG.workerCount },
//     () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
// );

// Visual debug mode: single VisTestEpisodeManager for visualization
const manager = new VisTestEpisodeManager();
manager.start();

createDebugVisualization(document.body, manager);
