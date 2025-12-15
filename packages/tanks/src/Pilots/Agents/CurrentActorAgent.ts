import * as tf from '@tensorflow/tfjs';

import { clamp } from 'lodash';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { calculateActionReward, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { NetworkModelManager } from './NetworkModelManager.ts';
import { ColoredNoiseApprox } from '../../../../ml-common/ColoredNoiseApprox.ts';
import { ACTION_HEAD_DIMS } from '../../../../ml/src/Models/Create.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';

export type TankAgent<A = Partial<DownloadableAgent> & Partial<LearnableAgent>> = A & {
    tankEid: number;
    scheduleUpdateTankBehaviour(width: number, height: number, frame: number): void;
    applyUpdateTankBehaviour(): void;
}

export type DownloadableAgent = {
    dispose(): void;
    sync(): Promise<void>;
    isSynced(): boolean;
}

export type LearnableAgent = {
    train: boolean;
    dispose(): void;
    getNoise(): tf.Tensor[];
    getVersion(): number;
    getMemory(): undefined | AgentMemory;
    getMemoryBatch(rewardBias: number): undefined | AgentMemoryBatch;
    evaluateTankBehaviour(width: number, height: number, frame: number, successRatio: number): void;
}

const currentActorUpdater = NetworkModelManager(() => getNetwork(Model.Policy));

export class CurrentActorAgent implements TankAgent<DownloadableAgent & LearnableAgent> {
    private memory = new AgentMemory();
    private noise = new ColoredNoiseApprox(ACTION_HEAD_DIMS, randomRangeFloat(0, 1));

    private initialActionReward?: number;

    constructor(public readonly tankEid: number, public train: boolean) {
    }

    public getVersion(): number {
        const net = currentActorUpdater.getNetwork();
        return net != null
            ? getNetworkExpIteration(net)
            : 0;
    }

    public getNoise() {
        return this.noise.sample().map(t => t.mul(0.1));
    }

    public getMemory(): undefined | AgentMemory {
        return this.train ? this.memory : undefined;
    }

    public getMemoryBatch(rewardBias: number): undefined | AgentMemoryBatch {
        return this.train ? this.memory.getBatch(rewardBias) : undefined;
    }

    public dispose() {
        this.memory.dispose();
        this.noise.dispose();
    }

    public async sync() {
        await currentActorUpdater.updateNetwork();
    }

    public isSynced() {
        return currentActorUpdater.getNetwork() != null;
    }

    public scheduleUpdateTankBehaviour(
        width: number,
        height: number,
    ) {
        currentActorUpdater.schedule(width, height, this);
    }

    public applyUpdateTankBehaviour() {
        const result = currentActorUpdater.get(
            this,
            this.train ||
            // @ts-ignore
            globalThis.disableNoise !== true
        );
        if (result == null) return;
        
        applyActionToTank(this.tankEid, result.actions, false);

        if (this.train && result.logProb != null && result.logits != null) {
            this.memory.addFirstPart(
                result.state,
                result.actions,
                result.logits,
                result.logProb,
            );
            this.initialActionReward = calculateActionReward(this.tankEid);
        }
    }

    public evaluateTankBehaviour(
        width: number,
        height: number,
        frame: number,
    ) {
        if (!this.train || this.memory.size() === 0) return;

        const isDead = getTankHealth(this.tankEid) <= 0;
        // const version = this.getVersion();

        const stateRewardMultiplier = 0.5;//; - 0.5 * clamp(unlerp(0, LEARNING_STEPS * 0.4, version), 0, 1);
        const actionRewardMultiplier = 1;// clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.3, 1);

        const stateReward = 0
        //  calculateStateReward(
        //     this.tankEid,
        //     width,
        //     height,
        //     clamp(version / (LEARNING_STEPS * 0.2), 0, 1)
        // );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;
        
        const frameReward = getFramePenalty(frame);
 
        // it's not all reward, also we have final reward for lose/win in the end of episode
        const reward = clamp(
            stateReward * stateRewardMultiplier
            + actionReward * actionRewardMultiplier
            + frameReward,
            -10,
            +10
        );

        this.memory.updateSecondPart(reward, isDead);
    }
}
