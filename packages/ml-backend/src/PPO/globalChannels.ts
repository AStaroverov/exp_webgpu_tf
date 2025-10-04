import { createClient, RealtimeChannel, SupabaseClient } from '@supabase/supabase-js';
import { Observable, Subject } from 'rxjs';
import { AgentMemoryBatch } from '../../../ml-common/Memory';

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.warn('⚠️  Supabase credentials not set, global channels will not work');
}

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient {
    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
        console.info('✅ Supabase client initialized for backend channels');
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
                console.info(`✅ [Backend] Subscribed to ${channelName}`);
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

// Receive new batch notifications from frontend
export const episodeSampleChannel = createSupabaseChannel<{
    batchId: string;
    fileName: string;
    networkVersion: number;
    scenarioIndex: number;
    successRatio: number;
    timestamp: string;
}>('new-batch');

// Send queue size to frontend
export const queueSizeChannel = createSupabaseChannel<number>('queue-size');

// Send model version updates
export const modelVersionChannel = createSupabaseChannel<{ model: string, version: number }>('model-version');
