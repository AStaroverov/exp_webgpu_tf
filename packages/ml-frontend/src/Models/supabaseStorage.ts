/**
 * Supabase Storage integration for loading models
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as tf from '@tensorflow/tfjs';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
const SUPABASE_BUCKET = import.meta.env.VITE_SUPABASE_BUCKET || 'models';

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient | null {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
        console.warn('⚠️  Supabase credentials not set, model loading disabled');
        return null;
    }

    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
        console.info('✅ Supabase client initialized');
    }

    return supabase;
}

/**
 * Create TensorFlow.js IOHandler for loading models from Supabase Storage
 * @param modelName - name of the model (e.g., 'policy-model', 'value-model')
 * @param version - version number
 */
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
                const basePath = `v${version}-${modelName}`;

                // Get public URLs for model files
                const { data: modelJsonData } = client.storage
                    .from(SUPABASE_BUCKET)
                    .getPublicUrl(`${basePath}/model.json`);

                const { data: weightsData } = client.storage
                    .from(SUPABASE_BUCKET)
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

                // Return ModelArtifacts
                return {
                    modelTopology: modelJSON.modelTopology,
                    weightSpecs: modelJSON.weightsManifest[0].weights,
                    weightData: weightsBuffer,
                    format: modelJSON.format,
                    generatedBy: modelJSON.generatedBy,
                    convertedBy: modelJSON.convertedBy,
                    trainingConfig: modelJSON.trainingConfig,
                    userDefinedMetadata: modelJSON.userDefinedMetadata,
                };
            } catch (error) {
                console.error(`❌ Failed to load ${modelName} v${version} from Supabase:`, error);
                throw error;
            }
        },
    };
}

/**
 * Load model from Supabase Storage
 * @param modelName - name of the model
 * @param version - version number (default: 0 for latest)
 */
export async function loadModelFromSupabase(
    modelName: string,
    version: number = 0
): Promise<tf.LayersModel> {
    const handler = createSupabaseIOHandler(modelName, version);
    return tf.loadLayersModel(handler);
}

/**
 * Get public URL for model in Supabase Storage
 */
export function getModelPublicUrl(modelName: string, version: number): string | null {
    const client = getSupabaseClient();
    if (!client) return null;

    const { data } = client.storage
        .from(SUPABASE_BUCKET)
        .getPublicUrl(`v${version}-${modelName}/model.json`);

    return data.publicUrl;
}

/**
 * Check if Supabase is configured
 */
export function isSupabaseConfigured(): boolean {
    return Boolean(SUPABASE_URL && SUPABASE_KEY);
}

/**
 * Check latest version available in Supabase (simplified)
 * For now, always returns 0 (latest)
 * TODO: Implement proper version checking via Supabase Storage list
 */
export async function getLatestModelVersion(modelName: string): Promise<number> {
    // For MVP, we always use v0 (LAST_NETWORK_VERSION)
    // Later can implement listing bucket to find latest version
    return 0;
}
