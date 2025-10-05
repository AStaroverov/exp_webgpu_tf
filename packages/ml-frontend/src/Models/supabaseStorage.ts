import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../lib/throwingError.ts';
import { LAST_NETWORK_VERSION } from '../../../ml-backend/src/Models/def.ts';
import { DEFAULT_EXPERIMENT } from '../../../ml-common/config.ts';

const SUPABASE_URL = import.meta.env.SUPABASE_URL || throwingError('SUPABASE_URL not set');
const SUPABASE_PUBLICK_KEY = import.meta.env.SUPABASE_PUBLICK_KEY || throwingError('SUPABASE_KEY not set');
const SUPABASE_BUCKET_MODELS = import.meta.env.SUPABASE_BUCKET_MODELS || throwingError('SUPABASE_MODELS_BUCKET not set');

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient | null {
    if (!SUPABASE_URL || !SUPABASE_PUBLICK_KEY) {
        console.warn('⚠️  Supabase credentials not set, model loading disabled');
        return null;
    }

    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_PUBLICK_KEY);
        console.info('✅ Supabase client initialized');
    }

    return supabase;
}

export function createSupabaseIOHandler(
    modelName: string,
    version: number
): tf.io.IOHandler {
    return {
        async load(): Promise<tf.io.ModelArtifacts> {
            const client = getSupabaseClient();
            if (!client) {
                throw new Error('Supabase client not configured');
            }

            try {
                const basePath = `${DEFAULT_EXPERIMENT.expName}/v${version}-${modelName}`;

                // Get public URLs for model files
                const { data: modelJsonData } = client.storage
                    .from(SUPABASE_BUCKET_MODELS)
                    .getPublicUrl(`${basePath}/model.json`);

                const { data: weightsData } = client.storage
                    .from(SUPABASE_BUCKET_MODELS)
                    .getPublicUrl(`${basePath}/weights.bin`);

                const modelJsonUrl = modelJsonData.publicUrl;
                const weightsUrl = weightsData.publicUrl;

                // Fetch model.json
                const modelJsonResponse = await fetch(modelJsonUrl);
                if (!modelJsonResponse.ok) {
                    throw new Error(`Failed to fetch model.json: ${modelJsonResponse.statusText}`);
                }
                const modelJSON = await modelJsonResponse.json();

                // Fetch weights.bin
                const weightsResponse = await fetch(weightsUrl);
                if (!weightsResponse.ok) {
                    throw new Error(`Failed to fetch weights.bin: ${weightsResponse.statusText}`);
                }
                const weightsBuffer = await weightsResponse.arrayBuffer();

                console.info(`✅ Loaded ${modelName} v${version} from Supabase`);

                return {
                    ...modelJSON,
                    weightData: weightsBuffer,
                };
            } catch (error) {
                console.error(`❌ Failed to load ${modelName} v${version} from Supabase:`, error);
                throw error;
            }
        },
    };
}

export async function loadModelFromSupabase(
    modelName: string,
    version: number = 0
): Promise<tf.LayersModel> {
    const handler = createSupabaseIOHandler(modelName, version);
    return tf.loadLayersModel(handler);
}

export function isSupabaseConfigured(): boolean {
    return Boolean(SUPABASE_URL && SUPABASE_PUBLICK_KEY);
}

export async function getAvailableVersions(modelName: string): Promise<number[]> {
    const client = getSupabaseClient();
    if (!client) {
        return [LAST_NETWORK_VERSION];
    }

    try {
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_MODELS)
            .list(DEFAULT_EXPERIMENT.expName, {
                limit: 5,
                offset: 0,
                sortBy: { column: 'created_at', order: 'desc' },
            });

        if (error) {
            console.error('Failed to list models:', error);
            return [LAST_NETWORK_VERSION];
        }

        // Find all directories matching v{number}-{modelName}
        const versions: number[] = [];
        const prefix = `-${modelName}`;

        for (const item of data) {
            if (item.name.startsWith('v') && item.name.endsWith(prefix)) {
                const versionStr = item.name.slice(1, item.name.indexOf('-'));
                const version = parseInt(versionStr);
                if (!isNaN(version) && version !== LAST_NETWORK_VERSION) {
                    versions.push(version);
                }
            }
        }

        // Always include latest version
        if (!versions.includes(LAST_NETWORK_VERSION)) {
            versions.push(LAST_NETWORK_VERSION);
        }

        return versions.sort((a, b) => b - a); // Sort descending (newest first)
    } catch (error) {
        console.error('Error getting available versions:', error);
        return [LAST_NETWORK_VERSION];
    }
}
