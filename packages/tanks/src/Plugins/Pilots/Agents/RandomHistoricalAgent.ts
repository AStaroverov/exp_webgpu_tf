import { applyActionToTank } from '../../../../../ppo_tanks/src/state/applyActionToTank.ts';
import { Model } from '../../../../../ppo/src/models/def.ts';
import { getRandomHistoricalNetwork } from '../../../../../ppo/src/models/storage.ts';
import { CONFIG } from '../../../../../ppo_tanks/src/config.ts';
import { createNetworkModelManager } from './NetworkModelManager.ts';
import { DownloadableAgent, TankAgent } from './CurrentActorAgent.ts';

const historcalActorUpdater = createNetworkModelManager(() => getRandomHistoricalNetwork(Model.Policy, CONFIG.savePath));

export class RandomHistoricalAgent implements TankAgent<DownloadableAgent> {
    constructor(public readonly tankEid: number) {
    }

    scheduleUpdateTankBehaviour(width: number, height: number): void {
        historcalActorUpdater.schedule(width, height, this);
    }

    applyUpdateTankBehaviour(): void {
        const result = historcalActorUpdater.get(this, false);
        if (result == null) return;
    
        applyActionToTank(this.tankEid, result.actions, false);
    }

    public dispose() {
        // Nothing to dispose
    }

    public async sync() {
        await historcalActorUpdater.updateNetwork();
    }

    public isSynced() {
        return historcalActorUpdater.getNetwork() != null;
    }
}
