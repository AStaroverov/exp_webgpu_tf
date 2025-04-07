export const reloadChannel = new BroadcastChannel('reload');

if (globalThis.document != null) {
    // we cannot listen reloadChannel, because don't send message at the same thread
    new BroadcastChannel('reload').onmessage = () => {
        globalThis.location.reload();
    };
}

export const learningRateChannel = new BroadcastChannel('learningRate');

export const metricsChannels = {
    rewards: new BroadcastChannel('rewards'),
    values: new BroadcastChannel('value'),
    returns: new BroadcastChannel('returns'),
    advantages: new BroadcastChannel('advantages'),
    kl: new BroadcastChannel('kl'),
    lr: new BroadcastChannel('lr'),
    valueLoss: new BroadcastChannel('valueLoss'),
    policyLoss: new BroadcastChannel('policyLoss'),
    trainTime: new BroadcastChannel('trainTime'),
    waitTime: new BroadcastChannel('waitTime'),
    versionDelta: new BroadcastChannel('versionDelta'),
    batchSize: new BroadcastChannel('batchSize'),
};