/**
 * Ably channels for communication with ml-backend
 */

import Ably from 'ably';
import { Observable, Subject } from 'rxjs';
import { AgentMemoryBatch } from '../../ml-common/Memory.ts';

const ABLY_API_KEY = import.meta.env.ABLY_API_KEY || '';

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

    channel.subscribe((message: Ably.Types.Message) => {
        subject.next(message.data as T);
    });

    return {
        obs: subject.asObservable(),
        emit: (data: T) => {
            channel.publish('message', data);
        }
    };
}

// Channel to send experience to ml-backend
export const episodeSampleChannel = createAblyChannel<EpisodeSample>('ml:experience.v1');

// Channel to receive queue size from ml-backend (for adaptive generation)
export const queueSizeChannel = createAblyChannel<number>('ml:queue-size.v1');

// Channel to receive curriculum state from ml-backend
export const curriculumStateChannel = createAblyChannel<CurriculumState>('ml:curriculum.v1');
