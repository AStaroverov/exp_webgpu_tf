import { createClient, RealtimeChannel, SupabaseClient } from '@supabase/supabase-js';
import { Observable, Subject } from 'rxjs';
import { throwingError } from '../../../lib/throwingError.ts';
import { AgentMemoryBatch } from '../../ml-common/Memory.ts';

const SUPABASE_URL = import.meta.env.SUPABASE_URL || throwingError('SUPABASE_URL not set');
const SUPABASE_PUBLICK_KEY = import.meta.env.SUPABASE_PUBLICK_KEY || throwingError('SUPABASE_KEY not set');
const SUPABASE_BUCKET_EXP_BATCHES = import.meta.env.SUPABASE_BUCKET_EXP_BATCHES || throwingError('SUPABASE_BUCKET_EXP_BATCHES not set');

if (!SUPABASE_URL || !SUPABASE_PUBLICK_KEY) {
    console.warn('⚠️  Supabase credentials not set, channels will not work');
}

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient {
    if (!supabase) {
        supabase = createClient(SUPABASE_URL!, SUPABASE_PUBLICK_KEY!);
        console.info('✅ Supabase client initialized for channels');
    }
    return supabase;
}

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

function createSupabaseChannel<T>(channelName: string): {
    obs: Observable<T>,
    emit: (data: T) => Promise<void>
} {
    const client = getSupabaseClient();
    const channel: RealtimeChannel = client.channel(channelName);
    const subject = new Subject<T>();

    // Subscribe to broadcasts
    channel
        .on('broadcast', { event: 'message' }, (payload) => {
            subject.next(payload.payload as T);
        })
        .subscribe((status) => {
            if (status === 'SUBSCRIBED') {
                console.info(`✅ Subscribed to ${channelName}`);
            }
        });

    return {
        obs: subject.asObservable(),
        emit: async (data: T) => {
            await channel.send({
                type: 'broadcast',
                event: 'message',
                payload: data,
            });
        }
    };
}

// Channel to receive queue size from ml-backend (for adaptive generation)
export const queueSizeChannel = createSupabaseChannel<number>('queue-size');

// Channel to send new batch notifications to ml-backend
export const newBatchChannel = createSupabaseChannel<{
    batchId: string;
    fileName: string;
    networkVersion: number;
    scenarioIndex: number;
    successRatio: number;
    timestamp: string;
}>('new-batch');

/**
 * Upload curriculum state to Supabase Storage
 * @param curriculumState - curriculum state data
 */
export async function uploadCurriculumState(curriculumState: CurriculumState): Promise<void> {
    const client = getSupabaseClient();

    try {
        const fileName = 'curriculumState.json';
        const data = JSON.stringify(curriculumState);

        // Upload to Supabase Storage (upsert to overwrite existing file)
        const { error: uploadError } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .upload(fileName, data, {
                contentType: 'application/json',
                upsert: true,
            });

        if (uploadError) {
            throw uploadError;
        }

        console.info(`✅ Uploaded curriculum state: version ${curriculumState.currentVersion}`);
    } catch (error) {
        console.error('❌ Failed to upload curriculum state:', error);
        throw error;
    }
}

/**
 * Download curriculum state from Supabase Storage
 * @returns curriculum state or default if not found
 */
export async function downloadCurriculumState(): Promise<CurriculumState> {
    const client = getSupabaseClient();
    const defaultState: CurriculumState = {
        currentVersion: 0,
        mapScenarioIndexToSuccessRatio: {},
    };

    try {
        const fileName = 'curriculumState.json';

        // Download from Supabase Storage
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .download(fileName);

        if (error) {
            console.warn('⚠️  Failed to download curriculum state, using default:', error.message);
            return defaultState;
        }

        if (!data) {
            console.warn('⚠️  No curriculum state found, using default');
            return defaultState;
        }

        const text = await data.text();
        const curriculumState = JSON.parse(text) as CurriculumState;

        console.info(`✅ Downloaded curriculum state: version ${curriculumState.currentVersion}`);
        return curriculumState;
    } catch (error) {
        console.error('❌ Failed to download curriculum state, using default:', error);
        return defaultState;
    }
}

/**
 * Upload episode sample to Supabase Storage and notify via Realtime
 * @param episodeSample - episode data with memory batch
 * @returns batch ID that was uploaded
 */
export async function uploadEpisodeSample(episodeSample: EpisodeSample): Promise<string> {
    const client = getSupabaseClient();

    try {
        // Generate unique batch ID
        const batchId = crypto.randomUUID();
        const fileName = `${batchId}.json`;

        // Prepare data for upload
        const data = {
            batchId,
            networkVersion: episodeSample.networkVersion,
            scenarioIndex: episodeSample.scenarioIndex,
            successRatio: episodeSample.successRatio,
            memoryBatch: {
                size: episodeSample.memoryBatch.size,
                states: episodeSample.memoryBatch.states,
                actions: episodeSample.memoryBatch.actions.map(a => Array.from(a)),
                mean: episodeSample.memoryBatch.mean.map(m => Array.from(m)),
                logStd: episodeSample.memoryBatch.logStd.map(ls => Array.from(ls)),
                logProbs: Array.from(episodeSample.memoryBatch.logProbs),
                rewards: Array.from(episodeSample.memoryBatch.rewards),
                dones: Array.from(episodeSample.memoryBatch.dones),
            },
            timestamp: new Date().toISOString(),
        };

        // Upload to Supabase Storage
        const { error: uploadError } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .upload(fileName, JSON.stringify(data), {
                contentType: 'application/json',
                upsert: false,
            });

        if (uploadError) {
            throw uploadError;
        }

        console.info(`✅ Uploaded experience batch: ${batchId}`);

        // Send notification via Realtime
        await newBatchChannel.emit({
            batchId,
            fileName,
            networkVersion: episodeSample.networkVersion,
            scenarioIndex: episodeSample.scenarioIndex,
            successRatio: episodeSample.successRatio,
            timestamp: data.timestamp,
        });

        return batchId;
    } catch (error) {
        console.error('❌ Failed to upload episode sample:', error);
        throw error;
    }
}
