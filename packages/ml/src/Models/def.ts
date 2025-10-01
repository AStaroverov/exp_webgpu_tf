export const LAST_NETWORK_VERSION = 0;

export enum Model {
    Policy = 'policy-model',
    Value = 'value-model',
}

export type NetworkInfo = {
    name: string,
    path: string,
    dateSaved: Date;
}