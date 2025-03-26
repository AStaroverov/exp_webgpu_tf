import { createDebugVisualization } from './debug.ts';
import { initSharedPPOSystem } from './init.ts';

initSharedPPOSystem().then(() => {
    createDebugVisualization(document.body);
}).catch(error => {
    console.error('Failed to initialize Shared Tank RL System:', error);
});