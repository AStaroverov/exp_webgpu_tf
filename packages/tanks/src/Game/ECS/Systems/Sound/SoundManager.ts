/**
 * Sound Manager - centralized audio management for the game
 * Handles loading, pooling, and playback of audio with spatial audio support
 */

export interface SoundConfig {
    src: string | string[];  // Single source or array for random selection
    maxInstances?: number;  // Max simultaneous plays of this sound
    volume?: number;
    loop?: boolean;
}

export interface PlayOptions {
    volume?: number;
    loop?: boolean;
    // For spatial audio
    x?: number;
    y?: number;
}

interface SoundInstance {
    audio: HTMLAudioElement;
    inUse: boolean;
    srcIndex: number;  // Which source this instance uses
}

class SoundManagerClass {
    private sounds: Map<string, SoundInstance[]> = new Map();
    private configs: Map<string, SoundConfig> = new Map();
    private masterVolume = 1;
    private enabled = true;

    // Listener position (camera/player)
    private listenerX = 0;
    private listenerY = 0;

    // Spatial audio settings
    private maxDistance = 1500;  // Beyond this, sound is silent
    private refDistance = 200;   // Distance at which volume is 1

    /**
     * Preload a sound for later use
     */
    async load(id: string, config: SoundConfig): Promise<void> {
        this.configs.set(id, config);

        const sources = Array.isArray(config.src) ? config.src : [config.src];
        const maxInstances = config.maxInstances ?? 3;
        const instances: SoundInstance[] = [];

        // Create instances, distributing across all sources
        for (let i = 0; i < maxInstances; i++) {
            const srcIndex = i % sources.length;
            const audio = new Audio(sources[srcIndex]);
            audio.volume = config.volume ?? 1;
            audio.loop = config.loop ?? false;

            // Preload
            audio.load();

            instances.push({ audio, inUse: false, srcIndex });
        }

        this.sounds.set(id, instances);
    }

    /**
     * Play a sound by id
     */
    play(id: string, options: PlayOptions = {}): HTMLAudioElement | null {
        if (!this.enabled) return null;

        const instances = this.sounds.get(id);
        const config = this.configs.get(id);

        if (!instances || !config) {
            console.warn(`Sound "${ id }" not loaded`);
            return null;
        }

        const sources = Array.isArray(config.src) ? config.src : [config.src];

        // For multiple sources, pick a random one and find matching available instance
        let instance: SoundInstance | undefined;
        if (sources.length > 1) {
            const randomSrcIndex = Math.floor(Math.random() * sources.length);
            // First try to find an available instance with the random source
            instance = instances.find(i => !i.inUse && i.srcIndex === randomSrcIndex);
            // If not found, try any available instance and change its source
            if (!instance) {
                instance = instances.find(i => !i.inUse);
                if (instance && instance.srcIndex !== randomSrcIndex) {
                    instance.audio.src = sources[randomSrcIndex];
                    instance.srcIndex = randomSrcIndex;
                }
            }
        } else {
            instance = instances.find(i => !i.inUse);
        }

        if (!instance) {
            // All instances busy - skip
            return null;
        }

        instance.inUse = true;
        const { audio } = instance;

        // Calculate volume based on distance
        let volume = options.volume ?? config.volume ?? 1;
        
        if (options.x !== undefined && options.y !== undefined) {
            const distance = this.getDistance(options.x, options.y);
            volume *= this.calculateDistanceAttenuation(distance);
        }

        audio.volume = volume * this.masterVolume;
        audio.loop = options.loop ?? config.loop ?? false;
        audio.currentTime = 0;
        
        audio.play().catch(() => {
            // Autoplay blocked, ignore
        });

        // Mark as available when done (unless looping)
        if (!audio.loop) {
            audio.onended = () => {
                instance.inUse = false;
            };
        }

        return audio;
    }

    /**
     * Stop all instances of a sound
     */
    stop(id: string): void {
        const instances = this.sounds.get(id);
        if (!instances) return;

        for (const instance of instances) {
            instance.audio.pause();
            instance.audio.currentTime = 0;
            instance.inUse = false;
        }
    }

    /**
     * Stop a specific audio instance
     */
    stopInstance(audio: HTMLAudioElement): void {
        audio.pause();
        audio.currentTime = 0;
        
        // Find and mark as not in use
        for (const instances of this.sounds.values()) {
            const instance = instances.find(i => i.audio === audio);
            if (instance) {
                instance.inUse = false;
                break;
            }
        }
    }

    /**
     * Update listener position (call every frame)
     */
    setListenerPosition(x: number, y: number): void {
        this.listenerX = x;
        this.listenerY = y;
    }

    /**
     * Get distance from listener to a point
     */
    private getDistance(x: number, y: number): number {
        const dx = x - this.listenerX;
        const dy = y - this.listenerY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate volume attenuation based on distance
     * Uses inverse distance model
     */
    private calculateDistanceAttenuation(distance: number): number {
        if (distance <= this.refDistance) return 1;
        if (distance >= this.maxDistance) return 0;

        // Linear rolloff
        const range = this.maxDistance - this.refDistance;
        const normalized = (distance - this.refDistance) / range;
        return 1 - normalized;
    }

    /**
     * Check if a position is within hearing range
     */
    isInRange(x: number, y: number): boolean {
        return this.getDistance(x, y) <= this.maxDistance;
    }

    setMasterVolume(volume: number): void {
        this.masterVolume = Math.max(0, Math.min(1, volume));
    }

    setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        if (!enabled) {
            this.stopAll();
        }
    }

    stopAll(): void {
        for (const [id] of this.sounds) {
            this.stop(id);
        }
    }

    dispose(): void {
        this.stopAll();
        this.sounds.clear();
        this.configs.clear();
    }
}

export const SoundManager = new SoundManagerClass();
