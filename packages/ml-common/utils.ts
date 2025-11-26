import * as tf from '@tensorflow/tfjs';
import { macroTasks } from '../../lib/TasksScheduler/macroTasks.ts';
import { ModelSettings } from '../ml/src/PPO/channels.ts';
import { CONFIG } from './config.ts';
import { CurriculumState, DEFAULT_CURRICULUM_STATE } from './Curriculum/types.ts';

export function setNetworkSettings(o: tf.LayersModel, settings: ModelSettings) {
    const meta = o.getUserDefinedMetadata() as ModelSettings;
    o.setUserDefinedMetadata({ ...meta, ...settings });
    setNetworkLearningRate(o, settings.lr ?? getNetworkLearningRate(o));
}

export function getNetworkSettings(network: tf.LayersModel): ModelSettings {
    const meta = network.getUserDefinedMetadata() as undefined | ModelSettings;
    return meta ?? {};
}

export function getNetworkExpIteration(network: tf.LayersModel): number {
    const meta = network.getUserDefinedMetadata() as undefined | { expIteration?: number };
    return meta?.expIteration ?? 0;
}

export function setNetworkLearningRate(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    return network.optimizer.learningRate ?? CONFIG.lrConfig.initial;
}

export function setNetworkCurriculumState(network: tf.LayersModel, curriculumState: CurriculumState) {
    network.setUserDefinedMetadata({
        ...network.getUserDefinedMetadata(),
        curriculumState,
    });
}

export function getNetworkCurriculumState(network: tf.LayersModel): CurriculumState {
    const meta = network.getUserDefinedMetadata() as undefined | { curriculumState?: CurriculumState };
    return cloneCurriculumState(meta?.curriculumState ?? DEFAULT_CURRICULUM_STATE);
}

function cloneCurriculumState(state: CurriculumState): CurriculumState {
    return {
        iteration: state.iteration,
        mapScenarioIndexToSuccessRatio: { ...state.mapScenarioIndexToSuccessRatio },
    };
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

