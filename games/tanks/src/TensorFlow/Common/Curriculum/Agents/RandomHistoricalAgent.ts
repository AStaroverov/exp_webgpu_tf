import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { act, MAX_STD_DEV } from '../../../PPO/train.ts';
import { prepareInputArrays } from '../../InputArrays.ts';
import { disposeNetwork, getRandomHistoricalNetwork } from '../../../Models/Utils.ts';
import { patientAction } from '../../utils.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { Model } from '../../../Models/def.ts';
import { TankAgent } from './CurrentActorAgent.ts';
import { clamp } from 'lodash-es';
import { lerp } from '../../../../../../../lib/math.ts';

export class RandomHistoricalAgent implements TankAgent {
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

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork!, state);

        applyActionToTank(
            this.tankEid,
            result.actions.map((v) => clamp(v / MAX_STD_DEV, -1, 1)),
            result.logStd.map((v) => lerp(0.3, 0.9, 1 - Math.exp(v) / MAX_STD_DEV)),
        );
    }

    private async load() {
        this.policyNetwork = await getRandomHistoricalNetwork(Model.Policy);
    }
}
