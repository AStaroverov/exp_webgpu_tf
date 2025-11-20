import * as tf from '@tensorflow/tfjs';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { patientAction } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getRandomHistoricalNetwork } from '../../../../ml/src/Models/Utils.ts';
import { pureAct } from '../../../../ml/src/PPO/train.ts';
import { DownloableAgent, TankAgent } from './CurrentActorAgent.ts';

export class RandomHistoricalAgent implements TankAgent<DownloableAgent> {
    private policyNetwork?: tf.LayersModel;

    constructor(public readonly tankEid: number) {
    }

    public dispose() {
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
    }

    public sync() {
        this.dispose();
        return patientAction(() => this.load());
    }

    public isSynced() {
        return this.policyNetwork != null;
    }

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        if (this.policyNetwork == null) return;

        const state = prepareInputArrays(this.tankEid, width, height);
        const result = pureAct(this.policyNetwork, state);

        applyActionToTank(this.tankEid, result.actions, false);
    }

    private async load() {
        this.policyNetwork = await getRandomHistoricalNetwork(Model.Policy);
    }
}
