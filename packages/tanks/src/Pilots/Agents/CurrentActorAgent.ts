import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { unlerp } from '../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { ColoredNoiseApprox } from '../../../../ml-common/ColoredNoiseApprox.ts';
import { LEARNING_STEPS } from '../../../../ml-common/consts.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration, patientAction } from '../../../../ml-common/utils.ts';
import { ACTION_HEAD_DIMS } from '../../../../ml/src/Models/Create.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { noisyAct, pureAct } from '../../../../ml/src/PPO/train.ts';
import { calculateActionReward, calculateStateReward, getDeathPenalty, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';

export type TankAgent<A = Partial<DownloableAgent> & Partial<LearnableAgent>> = A & {
    tankEid: number;
    updateTankBehaviour(width: number, height: number, frame: number): void;
}

export type DownloableAgent = {
    dispose(): void;
    sync(): Promise<void>;
    isSynced(): boolean;
}

export type LearnableAgent = {
    dispose(): void;
    getVersion(): number;
    getMemory(): undefined | AgentMemory;
    getMemoryBatch(gameOverReward: number): undefined | AgentMemoryBatch;
    evaluateTankBehaviour(width: number, height: number, frame: number): void;
}

export class CurrentActorAgent implements TankAgent<DownloableAgent & LearnableAgent> {
    private memory = new AgentMemory();
    private noise: ColoredNoiseApprox;
    private policyNetwork?: tf.LayersModel;

    private initialActionReward?: number;

    constructor(public readonly tankEid: number, private train: boolean) {
        this.noise = new ColoredNoiseApprox(ACTION_HEAD_DIMS, randomRangeFloat(0.3, 0.7));
    }

    public getVersion(): number {
        return this.policyNetwork != null ? getNetworkExpIteration(this.policyNetwork) : 0;
    }

    public getMemory(): undefined | AgentMemory {
        return this.train ? this.memory : undefined;
    }

    public getMemoryBatch(gameOverReward: number): undefined | AgentMemoryBatch {
        return this.train ? this.memory.getBatch(gameOverReward) : undefined;
    }

    public dispose() {
        disposeNetwork(this.policyNetwork);
        this.memory.dispose();
        this.noise?.dispose();
    }

    public async sync() {
        await patientAction(() => this.load());
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

        if (this.train) {
            const result = noisyAct(this.policyNetwork, state, this.noise.sample());

            applyActionToTank(this.tankEid, result.actions, false);
            this.memory.addFirstPart(
                state,
                result.actions,
                result.logits,
                result.logProb,
            );
            this.initialActionReward = calculateActionReward(this.tankEid);
        } else {
            // @ts-ignore
            const useNoise = globalThis.disableNoise !== true;
            const result = useNoise
                ? noisyAct(this.policyNetwork, state, this.noise.sample())
                : pureAct(this.policyNetwork, state);

            applyActionToTank(this.tankEid, result.actions, false);
        }
    }

    public evaluateTankBehaviour(
        width: number,
        height: number,
        frame: number,
    ) {
        if (!this.train || this.memory.size() === 0) return;

        const isDead = getTankHealth(this.tankEid) <= 0;
        const version = this.getVersion();
        const stateReward = calculateStateReward(
            this.tankEid,
            width,
            height,
            clamp(version / (LEARNING_STEPS * 0.2), 0, 1)
        );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;
        const stateRewardMultiplier = clamp(1 - unlerp(0, LEARNING_STEPS * 0.4, version), 0, 0.5);
        const actionRewardMultiplier = clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.3, 1);// - clamp(unlerp(LEARNING_STEPS * 0.6, LEARNING_STEPS, version), 0, 0.5);

        const frameReward = getFramePenalty(frame);
        const deathReward = getDeathPenalty(isDead);
        // it's not all reward, also we have final reward for lose/win in the end of episode
        const reward = clamp(
            stateReward * stateRewardMultiplier
            + actionReward * actionRewardMultiplier
            + frameReward
            + deathReward,
            -5,
            +5
        );

        this.memory.updateSecondPart(reward, isDead);
    }

    private async load() {
        this.policyNetwork = await getNetwork(Model.Policy);
    }
}
