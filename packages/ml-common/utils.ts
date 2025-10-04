import * as tf from '@tensorflow/tfjs';
import { macroTasks } from '../../lib/TasksScheduler/macroTasks.ts';
import { CONFIG } from './config.ts';

export function disposeNetwork(network: tf.LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}

export function setNetworkExpIteration(o: tf.LayersModel, it: number) {
    o.setUserDefinedMetadata({ ...o.getUserDefinedMetadata(), expIteration: it });
}

export function getNetworkExpIteration(network: tf.LayersModel): number {
    const meta = network.getUserDefinedMetadata() as { expIteration?: number } ?? { expIteration: 0 };
    return meta.expIteration ?? 0;
}

export function setNetworkLearningRate(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    return network.optimizer.learningRate ?? CONFIG.lrConfig.initial;
}

export async function patientAction<T>(action: () => T | Promise<T>, attempts: number = 100): Promise<T> {
    while (true) {
        attempts--;
        try {
            return await action();
        } catch (error) {
            if (attempts <= 0) {
                throw error;
            }

            await new Promise(resolve => macroTasks.addTimeout(resolve, 30));
        }
    }
}

