import Ably from 'ably';
import { Observable, Subject } from 'rxjs';
import { AgentMemoryBatch } from '../Common/Memory.ts';

const ABLY_API_KEY = process.env.ABLY_API_KEY || '';

if (!ABLY_API_KEY) {
    console.warn('⚠️  ABLY_API_KEY not set, global channels will not work');
}

const ably = new Ably.Realtime(ABLY_API_KEY);

export type EpisodeSample = {
    memoryBatch: AgentMemoryBatch,
    networkVersion: number,
    scenarioIndex: number,
    successRatio: number,
}

export type CurriculumState = {
    currentVersion: number,
    mapScenarioIndexToSuccessRatio: Record<number, number>,
}

function createAblyChannel<T>(channelName: string): {
    obs: Observable<T>,
    emit: (data: T) => void
} {
    const channel = ably.channels.get(channelName);
    const subject = new Subject<T>();


    channel.subscribe((message) => {
        subject.next(message.data as T);
    });

    return {
        obs: subject.asObservable(),
        emit: (data: T) => {
            channel.publish('message', data);
        }
    };
}

export const episodeSampleChannel = createAblyChannel<EpisodeSample>('ml:experience.v1');
export const queueSizeChannel = createAblyChannel<number>('ml:queue-size.v1');
export const curriculumStateChannel = createAblyChannel<CurriculumState>('ml:curriculum.v1');

export const modelVersionChannel = createAblyChannel<{ model: string, version: number }>('ml:model.weights.v1');
