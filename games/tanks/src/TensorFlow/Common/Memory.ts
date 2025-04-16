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

    addFirstPart(id: number, state: InputArrays, action: Float32Array, mean: Float32Array, logStd: Float32Array, logProb: number) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.addFirstPart(state, action, mean, logStd, logProb);
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
    private actions: Float32Array[] = [];
    private mean: Float32Array[] = [];
    private logStd: Float32Array[] = [];
    private logProbs: number[] = [];
    private rewards: number[] = [];
    private dones: boolean[] = [];

    private tmpRewards: number[] = [];
    private tmpDones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    addFirstPart(state: InputArrays, action: Float32Array, mean: Float32Array, logStd: Float32Array, logProb: number) {
        this.collapseTmpData();

        this.states.push(state);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        this.tmpRewards.push(reward);
        this.tmpDones.push(done);
    }

    getBatch(): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }
        if (this.tmpDones.length > 0 || this.tmpRewards.length > 0) {
            this.collapseTmpData();
        }
        if (this.states.length !== this.rewards.length || this.states.length !== this.dones.length) {
            const minLen = Math.min(this.states.length, this.rewards.length, this.dones.length);
            this.states.length = minLen;
            this.actions.length = minLen;
            this.mean.length = minLen;
            this.logStd.length = minLen;
            this.logProbs.length = minLen;
            this.rewards.length = minLen;
            this.dones.length = minLen;
        }

        const shapedRewards = rewardsShaping(this.rewards);
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
        this.rewards = [];
        this.dones = [];
        this.tmpRewards = [];
        this.tmpDones = [];
    }

    private collapseTmpData() {
        if (this.states.length > this.rewards.length) {
            this.rewards.push(this.tmpRewards.reduce((acc, r) => acc + r, 0));
            this.dones.push(this.tmpDones.reduce((acc, d) => acc || d, false));
            this.tmpRewards = [];
            this.tmpDones = [];
        }
    }
}

/**
 * Reward shaping idea:
 * r_t_prime = (r_t * k) + (r_t - r_{t-1})
 * r_t * k       — keeps a (small) absolute measure of how good the current
 *                 state is (good vs. bad), scaled down by factor k.
 * r_t - r_{t-1} — adds a delta term that rewards the agent for any
 *                 immediate improvement and penalises deterioration.
 *
 * In practice this lets the policy focus on *progress* (delta) while still
 * anchoring behaviour to an overall notion of “healthy” or “dangerous”
 * states.  Tune k so the delta dominates, but the absolute term prevents
 * agents from exploiting loops or camping in mediocre positions.
 */
function rewardsShaping(rewards: number[], k: number = 0.3): number[] {
    return rewards.map((reward, i) => {
        return reward * k + (i === 0 ? 0 : (reward - rewards[i - 1]));
    });
}