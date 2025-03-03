import { initSharedRLSystem } from './init.ts';
import { createDebugVisualization } from './debug.ts';

initSharedRLSystem().then(() => {
    createDebugVisualization(document.body);
}).catch(error => {
    console.error('Failed to initialize Shared Tank RL System:', error);
});