import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { getRandomHistoricalNetwork } from '../../../../ml/src/Models/Utils.ts';
import { NetworkModelManager } from './NetworkModelManager.ts';
import { DownloadableAgent, TankAgent } from './CurrentActorAgent.ts';

const historcalActorUpdater = NetworkModelManager(() => getRandomHistoricalNetwork(Model.Policy));

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
