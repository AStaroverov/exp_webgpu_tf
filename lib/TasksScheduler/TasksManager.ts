import {EMPTY_ARRAY} from '../const';
import {LinkedMap} from '../LinkedMap';
import {Opaque} from "../Types";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TCallback = (...args: any[]) => any;
export type TCallbackId = Opaque<'CallbackId', number>;
export type TCallbackMetadata = {
    cb: TCallback;
    ctx: unknown | null;
    args: unknown[];
    delta: number;
    delay: number;
    times: number;
    // Seed time and Time from last check
    lastTime: number;
};

export class TasksManager {
    private linkedMap = LinkedMap.create<TCallbackId, TCallbackMetadata>();
    private lastId = 0;

    private readonly destroyer: VoidFunction;

    constructor(ticker: (fn: VoidFunction) => VoidFunction) {
        this.destroyer = ticker(this.tick);
    }

    destroy(): void {
        this.destroyer();
    }

    createId(): TCallbackId {
        this.lastId = this.lastId === Number.MAX_SAFE_INTEGER ? 0 : this.lastId + 1;
        return this.lastId as TCallbackId;
    }

    addTask(
        cb: TCallback,
        props: {
            id?: number;
            ctx?: unknown;
            args?: unknown[];
            delay?: number;
            times?: number;
            insertFirst?: boolean;
        },
    ): VoidFunction {
        const id = (props.id ?? this.createId()) as TCallbackId;
        const meta: TCallbackMetadata = {
            cb,
            ctx: props.ctx ?? null,
            args: props.args ?? EMPTY_ARRAY,
            times: props.times ?? Infinity,
            delay: props.delay ?? 0,
            delta: props.delay ?? 0,
            lastTime: Date.now(),
        };

        if (props.insertFirst) {
            LinkedMap.insertFirst(this.linkedMap, id, meta);
        } else {
            LinkedMap.insertLast(this.linkedMap, id, meta);
        }

        return this.deleteTask.bind(this, id);
    }

    addInterval(
        cb: TCallback,
        delay: number,
        props?: {
            id?: number;
            ctx?: unknown;
            args?: unknown[];
            insertFirst?: boolean;
        },
    ): VoidFunction {
        return this.addTask(cb, {...props, delay, times: Infinity});
    }

    addTimeout(
        cb: TCallback,
        delay: number,
        props?: {
            id?: number;
            ctx?: unknown;
            args?: unknown[];
            insertFirst?: boolean;
        },
    ): VoidFunction {
        return this.addTask(cb, {...props, delay, times: 1});
    }

    deleteTask(id: number): void {
        LinkedMap.removeById(this.linkedMap, id);
    }

    protected runAllCallbacks(now: number): void {
        LinkedMap.forEach(this.linkedMap, (node) =>
            this.tryRunCallback(node.id, node.value, now, this.getDelta(now, node.value)),
        );
    }

    protected getDelta(now: number, meta: TCallbackMetadata): number {
        return now - meta.lastTime;
    }

    protected tryRunCallback(
        id: TCallbackId,
        meta: TCallbackMetadata,
        now: number,
        delta: number,
    ): void {
        meta.delta -= delta;
        meta.lastTime = now;

        if (meta.delta <= 0) {
            meta.times -= 1;

            try {
                meta.cb.apply(meta.ctx, meta.args);
            } catch (err) {
                console.error(err);
            }

            if (meta.times > 0) {
                meta.delta = meta.delay;
            } else {
                this.deleteTask(id);
            }
        }
    }

    private tick = (): void => {
        this.runAllCallbacks(Date.now());
    };
}
