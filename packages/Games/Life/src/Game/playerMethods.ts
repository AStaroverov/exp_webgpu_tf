import { initTensorFlow } from '../../../../ml-common/initTensorFlow.js';
import { destroyEngine } from './engine.js';
import { activateBots, deactivateBots, finalizeGameState } from './gameMethods.js';
import { toggleGame } from './modules/gameState.js';

export const startGame = async () => {
    await initTensorFlow('wasm');
    await finalizeGameState();
    activateBots();
    toggleGame(true);
};

export const exitGame = () => {
    deactivateBots();
    destroyEngine();
    toggleGame(false);
};
