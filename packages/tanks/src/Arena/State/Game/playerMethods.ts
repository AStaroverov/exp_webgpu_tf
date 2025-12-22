import { initTensorFlow } from '../../../../../ml-common/initTensorFlow.ts';
import { destroyEngine } from './engine.ts';
import { activateBots, deactivateBots, finalizeGameState } from './gameMethods.ts';
import { toggleGame } from './modules/gameState.ts';

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
