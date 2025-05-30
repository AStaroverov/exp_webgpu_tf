import * as tf from '@tensorflow/tfjs';
import { getNetworkVersion, patientAction } from '../../TensorFlow/Common/utils.ts';
import { disposeNetwork, getModelFromFS } from '../../TensorFlow/Models/Utils.ts';
import { prepareInputArrays } from '../../TensorFlow/Common/InputArrays.ts';
import { act, MAX_STD_DEV } from '../../TensorFlow/PPO/train.ts';
import { applyActionToTank } from '../../TensorFlow/Common/applyActionToTank.ts';
import { clamp } from 'lodash-es';
import { lerp } from '../../../../../lib/math.ts';
import { Model } from '../../TensorFlow/Models/def.ts';
import { TankAgent } from './CurrentActorAgent.ts';

export class VersionedAgent implements TankAgent {
    public readonly tankEid: number;
    public readonly path: string;
    private policyNetwork?: tf.LayersModel;

    constructor(tankEid: number, path: string) {
        this.path = path;
        this.tankEid = tankEid;

        void this.sync();
    }

    public isReady() {
        return this.policyNetwork != null;
    }

    public getVersion() {
        return this.policyNetwork != null ? getNetworkVersion(this.policyNetwork) : 0;
    }

    public dispose() {
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
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

    private async sync(): Promise<void> {
        this.dispose();
        await patientAction(() => this.load());
    }

    private async load() {
        this.policyNetwork = await getModelFromFS(Model.Policy, this.path);
    }
}
