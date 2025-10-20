import { InputArrays } from './InputArrays.ts';

// PPO Memory Batch - contains advantages and returns
export type AgentMemoryBatch = {
    size: number,
    states: InputArrays[],
    actions: Float32Array[],
    rewards: Float32Array,
    dones: Float32Array,
    mean: Float32Array[],
    logStd: Float32Array[],
    logProbs: Float32Array,
    perturbed: Float32Array, // 0 = not perturbed, 1 = perturbed
}

// SAC Memory Batch - contains next states for off-policy learning
// nextStates создаются из states[i+1] при подготовке батча
export type SACMemoryBatch = {
    size: number,
    states: InputArrays[],      // states[0..n-1]
    actions: Float32Array[],    // actions[0..n-1]
    rewards: Float32Array,      // rewards[0..n-1]
    nextStates: InputArrays[],  // states[1..n] - создаются из states
    dones: Float32Array,        // dones[0..n-1]
    perturbed: Float32Array,    // 0 = not perturbed, 1 = perturbed
}

export class AgentMemory {
    public states: InputArrays[] = [];
    public actions: Float32Array[] = [];
    public mean: Float32Array[] = [];
    public logStd: Float32Array[] = [];
    public logProbs: number[] = [];
    public rewards: number[] = [];
    public dones: boolean[] = [];
    public perturbed: boolean = false;

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
        logStd: Float32Array,
        logProb: number,
    ) {
        if (this.isDone()) return;
        if (this.states.length !== this.rewards.length) return;

        this.states.push(state);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
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

        // Создаем массив perturbed для каждого сэмпла
        const perturbedArray = new Float32Array(this.states.length);
        perturbedArray.fill(this.perturbed ? 1.0 : 0.0);

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logStd: (this.logStd),
            logProbs: new Float32Array(this.logProbs),
            rewards: rewards,
            dones: dones,
            perturbed: perturbedArray,
        };
    }

    /**
     * Get SAC batch - creates nextStates from states[i+1]
     * Обрезает все массивы на 1, чтобы для каждого state был nextState
     */
    getSACBatch(gameOverReward = 0): undefined | SACMemoryBatch {
        this.setMinLength();

        if (this.states.length === 0) {
            return undefined;
        }

        // Обрезаем массивы на 1, чтобы для каждого s[i] был s[i+1]
        const size = this.states.length - 1;

        if (size <= 0) {
            return undefined;
        }

        const states = this.states.slice(0, size);           // [0..n-1]
        const actions = this.actions.slice(0, size);         // [0..n-1]
        const nextStates = this.states.slice(1);             // [1..n]

        const rewards = new Float32Array(this.rewards.slice(0, size));
        rewards[rewards.length - 1] += gameOverReward;

        // @ts-expect-error - js conversion
        const dones = new Float32Array(this.dones.slice(0, size));
        dones[dones.length - 1] = 1.0;

        // Create perturbed array for each sample
        const perturbedArray = new Float32Array(size);
        perturbedArray.fill(this.perturbed ? 1.0 : 0.0);

        return {
            size,
            states,
            actions,
            rewards,
            nextStates,
            dones,
            perturbed: perturbedArray,
        };
    }

    dispose() {
        this.states.length = 0;
        this.actions.length = 0;
        this.mean.length = 0;
        this.logStd.length = 0;
        this.logProbs.length = 0;
        this.rewards.length = 0;
        this.dones.length = 0;
        this.perturbed = false;
    }

    private setMinLength() {
        const minLength = Math.min(
            this.states.length,
            this.actions.length,
            this.mean.length,
            this.logStd.length,
            this.logProbs.length,
            this.rewards.length,
            this.dones.length,
        );

        this.states.length = minLength;
        this.actions.length = minLength;
        this.mean.length = minLength;
        this.logStd.length = minLength;
        this.logProbs.length = minLength;
        this.rewards.length = minLength;
        this.dones.length = minLength;
    }
}

// SAC Memory - stores transitions (s, a, r, done)
// nextStates создаются автоматически из states[i+1]
export class SACMemory {
    public states: InputArrays[] = [];
    public actions: Float32Array[] = [];
    public rewards: number[] = [];
    public dones: boolean[] = [];
    public perturbed: boolean = false;

    constructor() {
    }

    size() {
        return this.states.length;
    }

    isDone() {
        return this.dones.length > 0 && this.dones[this.dones.length - 1];
    }

    /**
     * Add a transition (s, a, r, done)
     * nextState будет взят из следующего state автоматически
     */
    addTransition(
        state: InputArrays,
        action: Float32Array,
        reward: number,
        done: boolean,
    ) {
        if (this.isDone()) return;

        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
        this.dones.push(done);
    }

    getBatch(gameOverReward = 0): undefined | SACMemoryBatch {
        this.setMinLength();

        if (this.states.length === 0) {
            return undefined;
        }

        // Обрезаем массивы на 1, чтобы для каждого s[i] был s[i+1]
        const size = this.states.length - 1;

        if (size <= 0) {
            return undefined;
        }

        const states = this.states.slice(0, size);           // [0..n-1]
        const actions = this.actions.slice(0, size);         // [0..n-1]
        const nextStates = this.states.slice(1);             // [1..n]

        const rewards = new Float32Array(this.rewards.slice(0, size));
        rewards[rewards.length - 1] += gameOverReward;

        // @ts-expect-error - js conversion
        const dones = new Float32Array(this.dones.slice(0, size));
        dones[dones.length - 1] = 1.0;

        // Create perturbed array for each sample
        const perturbedArray = new Float32Array(size);
        perturbedArray.fill(this.perturbed ? 1.0 : 0.0);

        return {
            size,
            states,
            actions,
            rewards,
            nextStates,
            dones,
            perturbed: perturbedArray,
        };
    }

    dispose() {
        this.states.length = 0;
        this.actions.length = 0;
        this.rewards.length = 0;
        this.dones.length = 0;
        this.perturbed = false;
    }

    private setMinLength() {
        const minLength = Math.min(
            this.states.length,
            this.actions.length,
            this.rewards.length,
            this.dones.length,
        );

        this.states.length = minLength;
        this.actions.length = minLength;
        this.rewards.length = minLength;
        this.dones.length = minLength;
    }
}
