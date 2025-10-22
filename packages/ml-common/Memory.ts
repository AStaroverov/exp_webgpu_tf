import { InputArrays } from './InputArrays.ts';

export type AgentMemoryBatch = {
    size: number,
    states: InputArrays[],
    actions: Float32Array[],
    rewards: Float32Array,
    dones: Float32Array,
    mean: Float32Array[],
    logProbs: Float32Array,
}

export class AgentMemory {
    public states: InputArrays[] = [];
    public actions: Float32Array[] = [];
    public mean: Float32Array[] = [];
    public logProbs: number[] = [];
    public rewards: number[] = [];
    public dones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    isDone() {
        return this.dones[this.dones.length - 1];
    }

    addFirstPart(
        state: InputArrays,
        action: Float32Array,
        mean: Float32Array,
        logProb: number,
    ) {
        if (this.isDone()) return;
        if (this.states.length !== this.rewards.length) return;

        this.states.push(state);
        this.actions.push(action);
        this.mean.push(mean);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        if (this.isDone()) return;
        if ((this.states.length - 1) !== this.rewards.length) return;

        this.rewards.push(reward);
        this.dones.push(done);
    }

    getBatch(gameOverReward = 0): undefined | AgentMemoryBatch {
        this.setMinLength();

        if (this.states.length === 0) {
            return undefined;
        }

        const rewards = new Float32Array(this.rewards)
        rewards[rewards.length - 1] += gameOverReward;

        // @ts-expect-error - js convertion
        const dones = new Float32Array(this.dones);
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logProbs: new Float32Array(this.logProbs),
            rewards: rewards,
            dones: dones,
        };
    }

    dispose() {
        this.states.length = 0;
        this.actions.length = 0;
        this.mean.length = 0;
        this.logProbs.length = 0;
        this.rewards.length = 0;
        this.dones.length = 0;
    }

    private setMinLength() {
        const minLength = Math.min(
            this.states.length,
            this.actions.length,
            this.mean.length,
            this.logProbs.length,
            this.rewards.length,
            this.dones.length,
        );

        this.states.length = minLength;
        this.actions.length = minLength;
        this.mean.length = minLength;
        this.logProbs.length = minLength;
        this.rewards.length = minLength;
        this.dones.length = minLength;
    }
}
