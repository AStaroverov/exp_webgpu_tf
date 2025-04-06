// import './node/init.ts';
import { createDebugVisualization } from '../Common/debug.ts';
import { setConsolePrefix } from '../Common/console.ts';
import { PlayerManager } from './Player/PlayerManager.ts';
import { setBackend } from '../Common/initTensorFlow.ts';

setConsolePrefix(`[MAIN]`);

// Main initialization function
async function initSystem() {
    // new Worker('./PolicyLearnerWorker.ts');
    // new Worker('./ValueLearnerWorker.ts');
    // Array.from(
    //     { length: getCurrentConfig().workerCount },
    //     () => new Worker('./ActorWorker.ts'),
    // );

    await setBackend('node');

    const playerManager = await PlayerManager.create();
    playerManager.start();

    return { manager: playerManager };
}

initSystem().then(({ manager }) => {
    createDebugVisualization(document.body, manager);
});