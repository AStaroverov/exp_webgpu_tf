import * as tf from '@tensorflow/tfjs';
import { get, isNumber } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { CurriculumState } from '../../../../ml-common/Curriculum/types.ts';
import { metricsChannels } from '../../../../ml-common/channels.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { getNetworkCurriculumState, getNetworkExpIteration, patientAction, setNetworkCurriculumState, setNetworkSettings } from '../../../../ml-common/utils.ts';
import { saveNetworkToDB } from '../../Models/Transfer.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';
import { Model } from '../../Models/def.ts';
import { curriculumStateChannel, EpisodeSample, episodeSampleChannel, learnProcessChannel, modelSettingsChannel } from '../channels.ts';
import { networkHealthCheck } from '../train.ts';
import { LearnData } from './createLearnerManager.ts';

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnData) => unknown | Promise<unknown>,
}) {
    let network = await getNetwork(modelName, () => {
        const newNetwork = createNetwork();
        patientAction(() => saveNetworkToDB(newNetwork, modelName));
        return newNetwork;
    });

    modelSettingsChannel.obs.subscribe((settings) => {
        if (isNumber(settings.lr)) {
            settings.lr = settings.lr * (modelName === Model.Value ? CONFIG.valueLRCoeff : 1);
        }
        setNetworkSettings(network, settings);
    });

    learnProcessChannel.response(async (batch: LearnData) => {
        try {
            await trainNetwork(network, batch);
            await patientAction(() => networkHealthCheck(network));
            await patientAction(() => saveNetworkToDB(network, modelName));

            return { modelName: modelName, version: getNetworkExpIteration(network) };
        } catch (e: Error | unknown) {
            console.error(e);

            disposeNetwork(network);
            console.info('Load last network after error...');
            network = await patientAction(() => getNetwork(modelName));

            return { modelName: modelName, error: get(e, 'message') ?? 'Unknown error', restart: e instanceof Error && e.message.includes('mem') };
        }
    });

    if (modelName === Model.Policy) {
        const lastCurriculumState = getNetworkCurriculumState(network);
        lastCurriculumState && curriculumStateChannel.emit(lastCurriculumState);

        episodeSampleChannel.obs.subscribe((sample) => {
            metricsChannels.successRatio.postMessage([{
                scenarioIndex: sample.scenarioIndex,
                successRatio: sample.successRatio,
                isReference: sample.isReference
            }]);

            if (sample.isReference) {
                const curriculumState = computeCurriculumState(sample, getNetworkCurriculumState(network));

                curriculumStateChannel.emit(curriculumState);
                setNetworkCurriculumState(network, curriculumState);
            }
        })
    }
}

const mapScenarioIndexToSuccessRatio = new Map<number, RingBuffer<number>>;
const mapScenarioIndexToAvgSuccessRatio = new Map<number, number>;
const computeCurriculumState = (sample: EpisodeSample, prev?: CurriculumState): CurriculumState => {
    if (!mapScenarioIndexToSuccessRatio.has(sample.scenarioIndex)) {
        mapScenarioIndexToSuccessRatio.set(sample.scenarioIndex, new RingBuffer(30));
    }
    mapScenarioIndexToSuccessRatio.get(sample.scenarioIndex)!.add(sample.successRatio);

    for (const [scenarioIndex, successRatioHistory] of mapScenarioIndexToSuccessRatio) {
        const length = successRatioHistory.getBufferLength();
        const size = successRatioHistory.getSize();
        const ratio = length > size / 2
            ? successRatioHistory.toArray().reduce((acc, v) => acc + v, 0) / length
            : prev?.mapScenarioIndexToSuccessRatio[scenarioIndex] ?? 0;

        mapScenarioIndexToAvgSuccessRatio.set(scenarioIndex, ratio);
    }

    return {
        iteration: sample.maxNetworkVersion,
        mapScenarioIndexToSuccessRatio: {
            ...prev?.mapScenarioIndexToSuccessRatio,
            ...Object.fromEntries(mapScenarioIndexToAvgSuccessRatio)
        },
    };
};

