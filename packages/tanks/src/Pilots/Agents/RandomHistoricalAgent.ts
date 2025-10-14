import * as tf from '@tensorflow/tfjs';
import { lerp } from '../../../../../lib/math.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { patientAction } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getRandomHistoricalNetwork } from '../../../../ml/src/Models/Utils.ts';
import { act, MAX_STD_DEV } from '../../../../ml/src/PPO/train.ts';
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
        const result = act(this.policyNetwork, state);

        applyActionToTank(
            this.tankEid,
            result.actions,
            result.logStd.map((v) => lerp(0.1, 1, 1 - Math.exp(v) / MAX_STD_DEV)),
        );
    }

    private async load() {
        this.policyNetwork = await getRandomHistoricalNetwork(Model.Policy);
    }
}
