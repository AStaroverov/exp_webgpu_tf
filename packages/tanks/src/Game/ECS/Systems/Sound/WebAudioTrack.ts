/**
 * WebAudioTrack - simple wrapper over Web Audio API for individual sound control
 * Provides play/pause/stop/resume/setVolume/setLoop functionality
 */

export type WebAudioTrackState = 'stopped' | 'playing' | 'paused';

export interface WebAudioTrackOptions {
    volume?: number;
    loop?: boolean;
    autoplay?: boolean;
}

export class WebAudioTrack {
    private ctx: AudioContext;
    private buffer: AudioBuffer | null = null;
    private source: AudioBufferSourceNode | null = null;
    private gainNode: GainNode;

    private _volume = 1;
    private _loop = false;
    private _state: WebAudioTrackState = 'stopped';

    // For pause/resume tracking
    private startTime = 0;
    private pausedAt = 0;

    // Callback for when playback ends naturally
    private _onEndedCallback: (() => void) | null = null;

    constructor(ctx: AudioContext, options: WebAudioTrackOptions = {}) {
        this.ctx = ctx;
        this._volume = options.volume ?? 1;
        this._loop = options.loop ?? false;

        this.gainNode = ctx.createGain();
        this.gainNode.gain.value = this._volume;
        this.gainNode.connect(ctx.destination);
    }

    /**
     * Load audio from URL
     */
    async load(url: string): Promise<this> {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        this.buffer = await this.ctx.decodeAudioData(arrayBuffer);
        return this;
    }

    /**
     * Load from existing AudioBuffer
     */
    setBuffer(buffer: AudioBuffer): this {
        this.buffer = buffer;
        return this;
    }

    /**
     * Connect to a different destination (e.g., GainNode for master volume)
     */
    connect(destination: AudioNode): this {
        this.gainNode.disconnect();
        this.gainNode.connect(destination);
        return this;
    }

    /**
     * Start playback from beginning
     */
    play(): this {
        if (!this.buffer) {
            console.warn('WebAudioTrack: No buffer loaded');
            return this;
        }

        // Stop any current playback
        this.stopSource();

        // Resume context if suspended
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        this.createAndStartSource(0);
        this._state = 'playing';
        this.pausedAt = 0;

        return this;
    }

    /**
     * Pause playback (can be resumed)
     */
    pause(): this {
        if (this._state !== 'playing' || !this.source) {
            return this;
        }

        // Calculate how far we've played
        this.pausedAt = (this.ctx.currentTime - this.startTime) % this.duration;
        this.stopSource();
        this._state = 'paused';

        return this;
    }

    /**
     * Resume from paused position
     */
    resume(): this {
        if (this._state !== 'paused' || !this.buffer) {
            return this;
        }

        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        this.createAndStartSource(this.pausedAt);
        this._state = 'playing';

        return this;
    }

    /**
     * Stop playback completely (resets position)
     */
    stop(): this {
        this.stopSource();
        this._state = 'stopped';
        this.pausedAt = 0;

        return this;
    }

    /**
     * Set volume (0 to 1)
     */
    setVolume(volume: number): this {
        this._volume = Math.max(0, Math.min(1, volume));
        this.gainNode.gain.value = this._volume;
        return this;
    }

    /**
     * Set loop mode
     */
    setLoop(loop: boolean): this {
        this._loop = loop;
        if (this.source) {
            this.source.loop = loop;
        }
        return this;
    }

    /**
     * Get current volume
     */
    get volume(): number {
        return this._volume;
    }

    /**
     * Get loop state
     */
    get loop(): boolean {
        return this._loop;
    }

    /**
     * Get current playback state
     */
    get state(): WebAudioTrackState {
        return this._state;
    }

    /**
     * Get buffer duration in seconds
     */
    get duration(): number {
        return this.buffer?.duration ?? 0;
    }

    /**
     * Get current playback position in seconds
     */
    get currentTime(): number {
        if (this._state === 'stopped') return 0;
        if (this._state === 'paused') return this.pausedAt;
        if (this._state === 'playing') {
            const elapsed = this.ctx.currentTime - this.startTime;
            return this._loop ? elapsed % this.duration : Math.min(elapsed, this.duration);
        }
        return 0;
    }

    /**
     * Seek to position (in seconds)
     */
    seek(time: number): this {
        if (!this.buffer) return this;

        const wasPlaying = this._state === 'playing';
        const position = Math.max(0, Math.min(time, this.duration));

        if (wasPlaying) {
            this.stopSource();
            this.createAndStartSource(position);
        } else {
            this.pausedAt = position;
        }

        return this;
    }

    /**
     * Check if audio is loaded
     */
    get isLoaded(): boolean {
        return this.buffer !== null;
    }

    /**
     * Register callback for when playback ends naturally (not on stop/pause)
     */
    onEnded(callback: (() => void) | null): this {
        this._onEndedCallback = callback;
        return this;
    }

    /**
     * Dispose and cleanup
     */
    dispose(): void {
        this.stopSource();
        this.gainNode.disconnect();
        this.buffer = null;
        this._onEndedCallback = null;
    }

    // --- Private methods ---

    private createAndStartSource(offset: number): void {
        if (!this.buffer) return;

        this.source = this.ctx.createBufferSource();
        this.source.buffer = this.buffer;
        this.source.loop = this._loop;
        this.source.connect(this.gainNode);

        this.source.onended = () => {
            if (this._state === 'playing' && !this._loop) {
                this._state = 'stopped';
                this.pausedAt = 0;
                this._onEndedCallback?.();
            }
        };

        this.startTime = this.ctx.currentTime - offset;
        this.source.start(0, offset);
    }

    private stopSource(): void {
        if (this.source) {
            try {
                this.source.stop();
            } catch {
                // Already stopped
            }
            this.source.disconnect();
            this.source.onended = null;
            this.source = null;
        }
    }
}

/**
 * Factory function - creates track with shared AudioContext
 */
let sharedContext: AudioContext | null = null;

export function createWebAudioTrack(options?: WebAudioTrackOptions): WebAudioTrack {
    if (!sharedContext) {
        sharedContext = new AudioContext();
    }
    return new WebAudioTrack(sharedContext, options);
}

/**
 * Get or create shared AudioContext
 */
export function getSharedAudioContext(): AudioContext {
    if (!sharedContext) {
        sharedContext = new AudioContext();
    }
    return sharedContext;
}

