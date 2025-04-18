// Буфер опыта для PPO
import { shuffle } from '../../../../../lib/shuffle.ts';
import { InputArrays } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';

export type Batch = {
    states: InputArrays[],
    actions: Float32Array[],
    mean: Float32Array[],
    logStd: Float32Array[],
    logProbs: Float32Array,
    // meta
    size: number,
    dones: Float32Array,
    rewards: Float32Array,
}

export class Memory {
    private map = new Map<number, SubMemory>();

    constructor() {
    }

    size() {
        let size = 0;

        this.map.forEach(subMemory => {
            size += subMemory.size();
        });

        return size;
    }

    toArray() {
        return Array.from(this.map.values());
    }

    addFirstPart(id: number, state: InputArrays, rewardState: number, action: Float32Array, mean: Float32Array, logStd: Float32Array, logProb: number) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.addFirstPart(state, rewardState, action, mean, logStd, logProb);
    }

    updateSecondPart(id: number, reward: number, done: boolean) {
        if (!this.map.has(id)) {
            throw new Error('SubMemory not found');
        }
        this.map.get(id)!.updateSecondPart(reward, done);
    }

    getBatch(): Batch {
        const batches = this.getBatches();
        const values = shuffle(Array.from(batches.values()));

        return {
            size: values.reduce((acc, batch) => acc + batch.size, 0),
            states: (values.map(batch => batch.states)).flat(),
            actions: (values.map(batch => batch.actions)).flat(),
            mean: (values.map(batch => batch.mean)).flat(),
            logStd: (values.map(batch => batch.logStd)).flat(),
            logProbs: flatTypedArray(values.map(batch => batch.logProbs)),
            rewards: flatTypedArray(values.map(batch => batch.rewards)),
            dones: flatTypedArray(values.map(batch => batch.dones)),
        };
    }

    getBatches() {
        const batches = new Map<number, Batch>();

        this.map.forEach((subMemory, id) => {
            batches.set(id, subMemory.getBatch());
        });

        return batches;
    }

    dispose() {
        this.map.forEach(subMemory => subMemory.dispose());
        this.map.clear();
    }
}

export class SubMemory {
    private states: InputArrays[] = [];
    private stateRewards: number[] = [];
    private actions: Float32Array[] = [];
    private mean: Float32Array[] = [];
    private logStd: Float32Array[] = [];
    private logProbs: number[] = [];
    private actionRewards: number[] = [];
    private dones: boolean[] = [];

    private tmpActionRewards: number[] = [];
    private tmpDones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    addFirstPart(state: InputArrays, stateReward: number, action: Float32Array, mean: Float32Array, logStd: Float32Array, logProb: number) {
        this.collapseTmpData();

        this.states.push(state);
        this.stateRewards.push(stateReward);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        this.tmpActionRewards.push(reward);
        this.tmpDones.push(done);
    }

    getBatch(): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }
        if (this.tmpDones.length > 0 || this.tmpActionRewards.length > 0) {
            this.collapseTmpData();
        }
        if (this.states.length !== this.actionRewards.length || this.states.length !== this.dones.length) {
            const minLen = Math.min(this.states.length, this.actionRewards.length, this.dones.length);
            this.states.length = minLen;
            this.actions.length = minLen;
            this.mean.length = minLen;
            this.logStd.length = minLen;
            this.logProbs.length = minLen;
            this.actionRewards.length = minLen;
            this.dones.length = minLen;
        }

        const shapedRewards = rewardsShaping(this.stateRewards, this.actionRewards);
        // The last reward is not shaped, because it is the reward for whole episode
        shapedRewards[shapedRewards.length - 1] = this.actionRewards[this.actionRewards.length - 1];

        const dones = this.dones.map(done => done ? 1.0 : 0.0);
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logStd: (this.logStd),
            logProbs: new Float32Array(this.logProbs),
            rewards: new Float32Array(shapedRewards),
            dones: new Float32Array(dones),
        };
    }

    dispose() {
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.actionRewards = [];
        this.dones = [];
        this.tmpActionRewards = [];
        this.tmpDones = [];
    }

    private collapseTmpData() {
        if (this.states.length > this.actionRewards.length) {
            const done = this.tmpDones.reduce((acc, d) => acc || d, false);

            // If the last action is done, we need to keep only the last reward for the whole episode
            if (done) {
                this.tmpActionRewards = this.tmpActionRewards.slice(-1);
            }

            const weights = rewardMultipliers(this.tmpActionRewards.length);
            const reward = this.tmpActionRewards.reduce((acc, rew, i) => acc + rew * weights[i], 0);

            this.dones.push(done);
            this.actionRewards.push(reward);
            this.tmpDones = [];
            this.tmpActionRewards = [];
        }
    }
}

function rewardMultipliers(limit: number): number[] {
    const weights = [];

    for (let i = 0; i < limit; i++) {
        weights.push(i + 1);
    }

    const sum = weights.reduce((a, b) => a + b, 0);
    const normalized = weights.map(w => w / sum);

    return normalized;
}

const REWARD_SHAPING_COEFFICIENT = 0.25;

function rewardsShaping(stateRewards: number[], actionRewards: number[], k = REWARD_SHAPING_COEFFICIENT): number[] {
    return actionRewards.map((reward, i) => {
        return reward * k + (reward - stateRewards[i]) * (1 - k);
    });
}