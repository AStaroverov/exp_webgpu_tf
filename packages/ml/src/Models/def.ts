export const LAST_NETWORK_VERSION = 0;

export enum Model {
    Policy = 'policy-model',
    Value = 'value-model',
    // SAC models
    Critic1 = 'critic1-model',
    Critic2 = 'critic2-model',
    TargetCritic1 = 'target-critic1-model',
    TargetCritic2 = 'target-critic2-model',
}

export type NetworkInfo = {
    name: string,
    path: string,
    dateSaved: Date;
}