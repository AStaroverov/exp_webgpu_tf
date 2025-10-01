import { loadLastNetwork, loadNetworkByPath, saveNetwork } from './Transfer.ts';
import { disposeNetwork } from './Utils.ts';
import { Model } from './def.ts';

/**
 * Restore models from DB or filesystem
 * Tries to load from DB first, then falls back to filesystem if provided
 * 
 * @param fallbackPath - Optional path to restore from if DB load fails (e.g., './assets/models/v1')
 */
export async function restoreModels(fallbackPath?: string): Promise<void> {
    try {
        // Try loading from DB (latest version)
        console.info('🔄 Loading models from DB...');
        const [policyNetwork, valueNetwork] = await Promise.all([
            loadLastNetwork(Model.Policy),
            loadLastNetwork(Model.Value),
        ]);

        if (!policyNetwork || !valueNetwork) {
            throw new Error('Models not found in DB');
        }

        console.info('✅ Models loaded from DB successfully');

        // Dispose immediately as they're already saved
        disposeNetwork(policyNetwork);
        disposeNetwork(valueNetwork);
    } catch (dbError) {
        console.warn('⚠️  Failed to load models from DB:', dbError);

        if (!fallbackPath) {
            throw new Error('No fallback path provided and DB load failed');
        }

        // Fallback: Load from filesystem and save to DB
        console.info(`🔄 Loading models from filesystem: ${fallbackPath}`);
        try {
            const [policyNetwork, valueNetwork] = await Promise.all([
                loadNetworkByPath(fallbackPath, Model.Policy),
                loadNetworkByPath(fallbackPath, Model.Value),
            ]);

            if (!policyNetwork || !valueNetwork) {
                throw new Error('Failed to load models from filesystem');
            }

            console.info('✅ Models loaded from filesystem');
            console.info('🔄 Saving models to DB...');

            // Save to DB for future use
            await Promise.all([
                saveNetwork(policyNetwork, Model.Policy),
                saveNetwork(valueNetwork, Model.Value),
            ]);

            console.info('✅ Models saved to DB');

            // Cleanup
            disposeNetwork(policyNetwork);
            disposeNetwork(valueNetwork);
        } catch (fsError) {
            console.error('❌ Failed to load models from filesystem:', fsError);
            throw new Error('Failed to restore models from both DB and filesystem');
        }
    }
}