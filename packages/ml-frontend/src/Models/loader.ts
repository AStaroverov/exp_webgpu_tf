/**
 * Model loading utilities
 * Adapts between filesystem loading (old) and Supabase loading (new)
 */

import * as tf from '@tensorflow/tfjs';
import { loadModelFromSupabase } from './supabaseStorage.ts';

/**
 * Load model - tries Supabase first, falls back to filesystem if needed
 * @param modelName - name of the model
 * @param version - version number (default: 0 for latest)
 */
export async function loadModel(
    modelName: string,
    version: number = 0
): Promise<tf.LayersModel> {
    // Try loading from Supabase
    console.info(`Loading ${modelName} v${version} from Supabase...`);
    return await loadModelFromSupabase(modelName, version);
}
