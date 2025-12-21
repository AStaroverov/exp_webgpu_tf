/**
 * Sound Configuration
 * 
 * Audio settings for game sounds and effects.
 */

export const SoundConfig = {
    /** Base volume for tank shooting */
    shootBaseVolume: 0.4,
    
    /** Additional volume per bullet width unit */
    shootVolumePerWidth: 0.085,
} as const;

export type SoundType = typeof SoundConfig;

