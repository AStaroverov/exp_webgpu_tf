import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { ColoredNoiseApprox } from '../../../../ml-common/ColoredNoiseApprox.ts';
import { InputArrays, prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration, patientAction } from '../../../../ml-common/utils.ts';
import { ACTION_HEAD_DIMS } from '../../../../ml/src/Models/Create.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { batchAct } from '../../../../ml/src/PPO/train.ts';
import { calculateActionReward, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';

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
    getNoise(): tf.Tensor[];
    dispose(): void;
    getVersion(): number;
    getMemory(): undefined | AgentMemory;
    getMemoryBatch(isWin: boolean): undefined | AgentMemoryBatch;
    evaluateTankBehaviour(width: number, height: number, frame: number, successRatio: number): void;
}

export const ActorUpdater = (getter: () => Promise<tf.LayersModel>) => {
    let network: undefined | tf.LayersModel = undefined;
    let promiseNetwork: undefined | Promise<tf.LayersModel> = undefined;
    let dateRequestNetwork: number = 0;

    let scheduledAgents: {width: number, height: number, agent: TankAgent}[] = []
    let computedAgents = new Map<TankAgent<unknown>, {
        state: InputArrays,
        actions: Float32Array,
        logits: Float32Array,
        logProb: number
    }>();

    const updateNetwork = async () => {
        const now = Date.now();
        const delta = now - dateRequestNetwork 
        
        if (delta > 10_000 || promiseNetwork == null) {
            network = undefined;
            promiseNetwork?.then(disposeNetwork);
            promiseNetwork = patientAction(getter).then((v) => (network = v));
            dateRequestNetwork = now;
        }

        return await promiseNetwork;
    }
    const getNetwork = () => network;
    
    const schedule = (width: number, height: number, agent: TankAgent<unknown>) => {
        if (computedAgents.size > 0) {
            computedAgents.clear();
        }

        scheduledAgents.push({
            width,
            height,
            agent
        });
    }
    
    const get = (agent: TankAgent<unknown>) => {
        if (network == null) {
            scheduledAgents = [];
            return;
        }

        if (scheduledAgents.length > 0) {
            const states = scheduledAgents.map(({width, height, agent}) => prepareInputArrays(agent.tankEid, width, height));
            const noises = scheduledAgents.map(({agent}) => (agent.train
                // @ts-ignore
                || globalThis.disableNoise !== true
            ) ? agent.getNoise?.() : undefined);
            const result = batchAct(network, states, noises);
            
            for (const [index, {agent}] of scheduledAgents.entries()) {
                computedAgents.set(agent, {
                    state: states[index],
                    ...result[index]
                });
            }

            scheduledAgents = [];
        }
        
        return computedAgents.get(agent);
    }

    return {
        updateNetwork,
        getNetwork,
        schedule,
        get,
    };
}

const currentActorUpdater = ActorUpdater(() => getNetwork(Model.Policy));

export class CurrentActorAgent implements TankAgent<DownloadableAgent & LearnableAgent> {
    private memory = new AgentMemory();
    private noise: ColoredNoiseApprox;

    private initialActionReward?: number;

    constructor(public readonly tankEid: number, public train: boolean) {
        this.noise = new ColoredNoiseApprox(ACTION_HEAD_DIMS, randomRangeFloat(0.3, 0.7));
    }

    public getNoise(): tf.Tensor[] {
        return this.noise.sample();
    }
    
    public getVersion(): number {
        const net = currentActorUpdater.getNetwork();
        return net != null
            ? getNetworkExpIteration(net)
            : 0;
    }

    public getMemory(): undefined | AgentMemory {
        return this.train ? this.memory : undefined;
    }

    public getMemoryBatch(isWin: boolean): undefined | AgentMemoryBatch {
        return this.train ? this.memory.getBatch(isWin) : undefined;
    }

    public dispose() {
        this.memory.dispose();
        this.noise?.dispose();
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
        const result = currentActorUpdater.get(this);
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

        const stateRewardMultiplier = 1;//; - 0.5 * clamp(unlerp(0, LEARNING_STEPS * 0.4, version), 0, 1);
        const actionRewardMultiplier = 1;// clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.3, 1);

        const stateReward = 0
        // calculateStateReward(
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
