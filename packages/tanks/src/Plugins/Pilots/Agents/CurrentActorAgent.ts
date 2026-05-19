import * as tf from '@tensorflow/tfjs';

import { clamp } from 'lodash';
import { applyActionToTank } from '../../../../../ppo_tanks/src/state/applyActionToTank.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../../ppo/src/memory/Memory.ts';
import { InputArrays } from '../../../../../ppo_tanks/src/state/InputArrays.ts';
import { getNetworkExpIteration } from '../../../../../ppo/src/models/networkMeta.ts';
import { Model } from '../../../../../ppo/src/models/def.ts';
import { getNetwork } from '../../../../../ppo/src/models/storage.ts';
import { CONFIG } from '../../../../../ppo_tanks/src/config.ts';
import { calculateActionReward, getFramePenalty } from '../../../../../ppo_tanks/src/reward/calculateReward.ts';
import { getTankHealth } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { createNetworkModelManager } from './NetworkModelManager.ts';
import { ACTION_HEAD_DIMS } from '../../../../../ppo_tanks/src/models/createTankNetworks.ts';
import { DirichletNoise } from '../../../../../ppo/src/noise/DirichletNoise.ts';

export type TankAgent<A = Partial<DownloadableAgent> & Partial<LearnableAgent>> = A & {
    tankEid: number;
    scheduleUpdateTankBehaviour(width: number, height: number, frame: number): void;
    applyUpdateTankBehaviour(width: number, height: number, frame: number): void;
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
    getMemory(): undefined | AgentMemory<InputArrays>;
    getMemoryBatch(rewardBias: number): undefined | AgentMemoryBatch<InputArrays>;
    evaluateTankBehaviour(width: number, height: number, frame: number): void;
}

const currentActorUpdater = createNetworkModelManager(() => getNetwork(Model.Policy, CONFIG.savePath));

export class CurrentActorAgent implements TankAgent<DownloadableAgent & LearnableAgent> {
    private memory = new AgentMemory<InputArrays>();
    private noise = new DirichletNoise(ACTION_HEAD_DIMS, {
        alpha: 0.3,  // α < 1 for sparse, peaked samples (high exploration)
        repeatInterval: [1, 6],
    });

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
        return this.noise.sample();
    }

    public getMemory(): undefined | AgentMemory<InputArrays> {
        return this.train ? this.memory : undefined;
    }

    public getMemoryBatch(finalReward: number): undefined | AgentMemoryBatch<InputArrays> {
        return this.train ? this.memory.getBatch(finalReward) : undefined;
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

    public applyUpdateTankBehaviour(_width: number, _height: number) {
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
        _width: number,
        _height: number,
        frame: number,
    ) {
        if (this.memory.size() === 0) return;

        const isDead = getTankHealth(this.tankEid) <= 0;
        const frameReward = getFramePenalty(frame);
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;

        const reward = clamp(
            (0
            + frameReward
            + actionReward
            ), -30, +30
        );

        this.memory.updateSecondPart(reward, isDead);
    }
}
