import { InputArrays } from './InputArrays.ts';

export type AgentMemoryBatch = {
    size: number,
    states: InputArrays[],
    actions: Float32Array[],
    rewards: Float32Array,
    dones: Float32Array,
    mean: Float32Array[],
    logStd: Float32Array[],
    logProbs: Float32Array,
}

export class AgentMemory {
    public states: InputArrays[] = [];
    public stateRewards: number[] = [];
    public actions: Float32Array[] = [];
    public mean: Float32Array[] = [];
    public logStd: Float32Array[] = [];
    public logProbs: number[] = [];
    public actionRewards: number[] = [];
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
        stateReward: number,
        action: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    ) {
        if (this.isDone()) {
            return;
        }
        this.states.push(state);
        this.stateRewards.push(stateReward);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        if (this.isDone()) {
            return;
        }
        this.actionRewards.push(reward);
        this.dones.push(done);
    }

    getBatch(): AgentMemoryBatch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }

        this.setMinLength();

        const rewards = this.actionRewards.map((aR, i) => aR - this.stateRewards[i]);
        const dones = this.dones.map(done => done ? 1.0 : 0.0);
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logStd: (this.logStd),
            logProbs: new Float32Array(this.logProbs),
            rewards: new Float32Array(rewards),
            dones: new Float32Array(dones),
        };
    }

    dispose() {
        this.states.length = 0;
        this.stateRewards.length = 0;
        this.actions.length = 0;
        this.mean.length = 0;
        this.logStd.length = 0;
        this.logProbs.length = 0;
        this.actionRewards.length = 0;
        this.dones.length = 0;
    }

    private setMinLength() {
        const minLength = Math.min(
            this.states.length,
            this.stateRewards.length,
            this.actions.length,
            this.mean.length,
            this.logStd.length,
            this.logProbs.length,
            this.actionRewards.length,
            this.dones.length,
        );

        this.states.length = minLength;
        this.stateRewards.length = minLength;
        this.actions.length = minLength;
        this.mean.length = minLength;
        this.logStd.length = minLength;
        this.logProbs.length = minLength;
        this.actionRewards.length = minLength;
        this.dones.length = minLength;
    }
}
