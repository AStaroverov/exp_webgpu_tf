import * as tf from '@tensorflow/tfjs';
import { lerp } from '../../../../../lib/math.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { getNetworkExpIteration, patientAction } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getRandomHistoricalNetwork } from '../../../../ml/src/Models/Utils.ts';
import { act } from '../../../../ml/src/PPO/train.ts';
import { DownloableAgent, TankAgent } from './CurrentActorAgent.ts';

export class RandomHistoricalAgent implements TankAgent<DownloableAgent> {
    private policyNetwork?: tf.LayersModel;
    private minLogStdDev?: number;
    private maxLogStdDev?: number;

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
        return this.policyNetwork != null && this.minLogStdDev != null && this.maxLogStdDev != null;
    }

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        if (!(this.policyNetwork != null && this.minLogStdDev != null && this.maxLogStdDev != null)) return;

        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork, state, this.minLogStdDev, this.maxLogStdDev);
        const maxStdDev = Math.exp(this.maxLogStdDev);

        applyActionToTank(
            this.tankEid,
            result.actions,
            result.logStd.map((v) => lerp(0.1, 1, 1 - Math.exp(v) / maxStdDev)),
        );
    }

    private async load() {
        this.policyNetwork = await getRandomHistoricalNetwork(Model.Policy);
        const iteration = getNetworkExpIteration(this.policyNetwork);
        this.minLogStdDev = CONFIG.minLogStdDev(iteration);
        this.maxLogStdDev = CONFIG.maxLogStdDev(iteration);
    }
}
