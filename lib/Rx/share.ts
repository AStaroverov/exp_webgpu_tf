import { ReplaySubject, share, shareReplay, timer } from 'rxjs';

export const shareReplayWithDelayedReset = <T>(delay: number, replay = 1) =>
    // almost same that inside shareReplay
    share<T>({
        connector: () => new ReplaySubject(replay),
        resetOnError: true,
        resetOnComplete: false,
        resetOnRefCountZero: () => timer(delay),
    });

export const shareReplayWithImmediateReset = <T>(replay = 1) =>
    shareReplay<T>({
        bufferSize: replay,
        refCount: true,
    });
