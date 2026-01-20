import { applyActionToTank } from '../../../../../ml-common/applyActionToTank.ts';
import { DownloadableAgent, TankAgent } from './CurrentActorAgent.js';
import { createNetworkModelManager, NetworkModelManager } from './NetworkModelManager.js';
import { Model } from '../../../../../ml/src/Models/def.ts';
import { loadNetworkFromFS } from '../../../../../ml/src/Models/Transfer.ts';

export class LoadedAgent implements TankAgent<DownloadableAgent> {
    constructor(
        public readonly tankEid: number,
        public readonly path: string,
        private readonly modelManager: NetworkModelManager
    ) {
    }
    
    scheduleUpdateTankBehaviour(width: number, height: number): void {
        this.modelManager.schedule(width, height, this);
    }

    applyUpdateTankBehaviour(): void {
        const result = this.modelManager.get(this, false);
        if (result == null) return;
    
        applyActionToTank(this.tankEid, result.actions, false);
    }

    public dispose() {
        // Nothing to dispose
    }

    public async sync() {
        await this.modelManager.updateNetwork();
    }

    public isSynced() {
        return this.modelManager.getNetwork() != null;
    }
}

// @TODO: mem leak, no dispose
const models = new Map<string, NetworkModelManager>();
export function getLoadedAgent(tankEid: number, path: string) {
    if (!models.has(path)) {
        const model = createNetworkModelManager(() => loadNetworkFromFS(path, Model.Policy));
        models.set(path, model);
    }

    return new LoadedAgent(tankEid, path, models.get(path)!);
}