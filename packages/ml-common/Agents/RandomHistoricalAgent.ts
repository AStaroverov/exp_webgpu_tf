import { clamp } from 'lodash-es';
import { lerp } from '../../../lib/math.ts';
import { Model } from '../../ml-backend/src/Models/def.ts';
import { act, MAX_STD_DEV } from '../../ml-backend/src/PPO/train.ts';
import { getRandomHistoricalNetwork } from '../../ml-frontend/src/Models/utils.ts';
import { applyActionToTank } from '../applyActionToTank.ts';
import { prepareInputArrays } from '../InputArrays.ts';
import * as tf from '../tf';
import { disposeNetwork, patientAction } from '../utils.ts';
import { DownloableAgent, TankAgent } from './types.ts';

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
        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork!, state);

        applyActionToTank(
            this.tankEid,
            result.actions.map((v) => clamp(v, -1, 1)),
            result.logStd.map((v) => lerp(0.1, 1, 1 - Math.exp(v) / MAX_STD_DEV)),
        );
    }

    private async load() {
        this.policyNetwork = await getRandomHistoricalNetwork(Model.Policy);
    }
}
