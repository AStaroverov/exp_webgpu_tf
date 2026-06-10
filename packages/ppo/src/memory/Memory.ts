export type AgentMemoryBatch<S> = {
  size: number;
  states: S[];
  actions: Float32Array[];
  rewards: Float32Array;
  dones: Float32Array;
  logits: Float32Array[];
  logProbs: Float32Array;
  masks?: Float32Array[];
};

export type PreparedBatch<S> = AgentMemoryBatch<S>;

export class AgentMemory<S> {
  public states: S[] = [];
  public actions: Float32Array[] = [];
  public logits: Float32Array[] = [];
  public masks: Float32Array[] = [];
  public logProbs: number[] = [];
  public rewards: number[] = [];
  public dones: boolean[] = [];

  constructor() {}

  size() {
    return this.states.length;
  }

  isDone() {
    return this.dones[this.dones.length - 1];
  }

  addFirstPart(
    state: S,
    action: Float32Array,
    logits: Float32Array,
    logProb: number,
    mask?: Float32Array,
  ) {
    if (this.isDone()) return;
    if (this.states.length !== this.rewards.length) return;

    this.states.push(state);
    this.actions.push(action);
    this.logits.push(logits);
    this.logProbs.push(logProb);
    if (mask != null) this.masks.push(mask);
  }

  updateSecondPart(reward: number, done: boolean) {
    if (this.isDone()) return;
    if (this.states.length - 1 !== this.rewards.length) return;

    this.rewards.push(reward);
    this.dones.push(done);
  }

  getBatch(finalReward: number): undefined | AgentMemoryBatch<S> {
    this.setMinLength();

    if (this.states.length === 0) {
      return undefined;
    }

    const rewards = new Float32Array(this.rewards);
    rewards[rewards.length - 1] += finalReward;

    // @ts-expect-error - js convertion
    const dones = new Float32Array(this.dones);
    dones[dones.length - 1] = 1.0;

    return {
      size: this.states.length,
      states: this.states.slice(),
      actions: this.actions.slice(),
      logits: this.logits.slice(),
      masks: this.masks.length > 0 ? this.masks.slice() : undefined,
      logProbs: new Float32Array(this.logProbs),
      rewards: rewards,
      dones: dones,
    };
  }

  dispose() {
    this.states.length = 0;
    this.actions.length = 0;
    this.logits.length = 0;
    this.masks.length = 0;
    this.logProbs.length = 0;
    this.rewards.length = 0;
    this.dones.length = 0;
  }

  private setMinLength() {
    // Masks are optional: an empty masks array (never-masked memory, e.g. ppo_tanks)
    // must NOT force minLength to 0 — only include it when populated.
    const lengths = [
      this.states.length,
      this.actions.length,
      this.logits.length,
      this.logProbs.length,
      this.rewards.length,
      this.dones.length,
    ];
    if (this.masks.length > 0) lengths.push(this.masks.length);

    const minLength = Math.min(...lengths);

    this.states.length = minLength;
    this.actions.length = minLength;
    this.logits.length = minLength;
    this.logProbs.length = minLength;
    this.rewards.length = minLength;
    this.dones.length = minLength;
    if (this.masks.length > 0) this.masks.length = minLength;
  }
}
