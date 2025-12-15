/**
 * Sound Manager - centralized audio management for the game
 * Uses Web Audio API (AudioContext) for low-latency playback with spatial audio support
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
    source: AudioBufferSourceNode | null;
    gainNode: GainNode;
    inUse: boolean;
    srcIndex: number;  // Which source this instance uses
}

interface LoadedSound {
    buffers: AudioBuffer[];  // One buffer per source
    instances: SoundInstance[];
}

class SoundManager {
    private ctx: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    
    private sounds: Map<string, LoadedSound> = new Map();
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
     * Initialize AudioContext (must be called after user interaction)
     */
    private ensureContext(): AudioContext {
        if (!this.ctx) {
            this.ctx = new AudioContext();
            this.masterGain = this.ctx.createGain();
            this.masterGain.gain.value = this.masterVolume;
            this.masterGain.connect(this.ctx.destination);
        }
        
        // Resume if suspended (browser autoplay policy)
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        
        return this.ctx;
    }

    /**
     * Preload a sound for later use
     */
    async load(id: string, config: SoundConfig): Promise<void> {
        const ctx = this.ensureContext();
        this.configs.set(id, config);

        const sources = Array.isArray(config.src) ? config.src : [config.src];
        const maxInstances = config.maxInstances ?? 3;

        // Load all audio buffers
        const buffers: AudioBuffer[] = await Promise.all(
            sources.map(async (src) => {
                const response = await fetch(src);
                const arrayBuffer = await response.arrayBuffer();
                return ctx.decodeAudioData(arrayBuffer);
            })
        );

        // Create gain nodes for instances (sources are created on play)
        const instances: SoundInstance[] = [];
        for (let i = 0; i < maxInstances; i++) {
            const gainNode = ctx.createGain();
            gainNode.gain.value = config.volume ?? 1;
            gainNode.connect(this.masterGain!);
            
            instances.push({
                source: null,
                gainNode,
                inUse: false,
                srcIndex: i % sources.length,
            });
        }

        this.sounds.set(id, { buffers, instances });
    }

    /**
     * Play a sound by id
     */
    play(id: string, options: PlayOptions = {}): AudioBufferSourceNode | null {
        if (!this.enabled) return null;

        const ctx = this.ensureContext();
        const sound = this.sounds.get(id);
        const config = this.configs.get(id);

        if (!sound || !config) {
            console.warn(`Sound "${id}" not loaded`);
            return null;
        }

        const { buffers, instances } = sound;

        // For multiple sources, pick a random one
        let instance: SoundInstance | undefined;
        const randomSrcIndex = buffers.length > 1 
            ? Math.floor(Math.random() * buffers.length) 
            : 0;

        // Find an available instance
        instance = instances.find(i => !i.inUse);

        if (!instance) {
            // All instances busy - skip
            return null;
        }

        instance.inUse = true;
        instance.srcIndex = randomSrcIndex;

        // Stop previous source if any
        if (instance.source) {
            try {
                instance.source.stop();
            } catch {
                // Already stopped
            }
            instance.source.disconnect();
        }

        // Create new source
        const source = ctx.createBufferSource();
        source.buffer = buffers[randomSrcIndex];
        source.loop = options.loop ?? config.loop ?? false;
        source.connect(instance.gainNode);

        // Calculate volume based on distance
        let volume = options.volume ?? config.volume ?? 1;
        
        if (options.x !== undefined && options.y !== undefined) {
            const distance = this.getDistance(options.x, options.y);
            volume *= this.calculateDistanceAttenuation(distance);
        }

        instance.gainNode.gain.value = volume;
        instance.source = source;

        source.start(0);

        // Mark as available when done (unless looping)
        if (!source.loop) {
            source.onended = () => {
                instance.inUse = false;
                instance.source = null;
            };
        }

        return source;
    }

    /**
     * Stop all instances of a sound
     */
    stop(id: string): void {
        const sound = this.sounds.get(id);
        if (!sound) return;

        for (const instance of sound.instances) {
            if (instance.source) {
                try {
                    instance.source.stop();
                } catch {
                    // Already stopped
                }
                instance.source.disconnect();
                instance.source = null;
            }
            instance.inUse = false;
        }
    }

    /**
     * Stop a specific audio instance
     */
    stopInstance(source: AudioBufferSourceNode): void {
        try {
            source.stop();
        } catch {
            // Already stopped
        }
        source.disconnect();
        
        // Find and mark as not in use
        for (const sound of this.sounds.values()) {
            const instance = sound.instances.find(i => i.source === source);
            if (instance) {
                instance.inUse = false;
                instance.source = null;
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
        if (this.masterGain) {
            this.masterGain.gain.value = this.masterVolume;
        }
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
        
        // Disconnect all gain nodes
        for (const sound of this.sounds.values()) {
            for (const instance of sound.instances) {
                instance.gainNode.disconnect();
            }
        }
        
        this.sounds.clear();
        this.configs.clear();
        
        if (this.ctx) {
            this.ctx.close();
            this.ctx = null;
            this.masterGain = null;
        }
    }
}

export const soundManager = new SoundManager();
