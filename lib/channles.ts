import { get } from 'lodash';
import { fromEvent, map, merge, Subject } from 'rxjs';

export function createChannel<Req, Res = unknown>(name: string, { global = false }: { global?: boolean } = {}) {
    const request = new Subject<Req>();
    const response = new Subject<Res>();
    const crossRequest = global ? new BroadcastChannel(name + '-request') : null;
    const crossResponse = global ? new BroadcastChannel(name + '-response') : null;

    const request$ = merge(
        request,
        crossRequest ? fromEvent(crossRequest, 'message').pipe(
            map((event: Event) => get(event, 'data') as Req),
        ) : [],
    );

    const response$ = merge(
        response,
        crossResponse ? fromEvent(crossResponse, 'message').pipe(
            map((event: Event) => get(event, 'data') as Res),
        ) : [],
    );

    return {
        emit: (data: Req) => {
            crossRequest?.postMessage(data);
            request.next(data);
        },
        obs: request$,
        request: (data: Req) => {
            crossRequest?.postMessage(data);
            request.next(data);
            return response$;
        },
        response: (cb: (data: Req) => Res | Promise<Res>) => {
            const sub = request$.subscribe(async (data) => {
                const res = await cb(data);
                response.next(res);
                crossResponse?.postMessage(res);
            });

            return () => sub.unsubscribe();
        },
    };
}
