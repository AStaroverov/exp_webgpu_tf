import { fromEvent, map, merge, Subject } from 'rxjs';
import { get } from 'lodash';

export function createChannel<Req, Res = unknown>(name: string) {
    const request = new Subject<Req>();
    const response = new Subject<Res>();
    const crossRequest = new BroadcastChannel(name + '-request');
    const crossResponse = new BroadcastChannel(name + '-response');

    const request$ = merge(
        request,
        fromEvent(crossRequest, 'message').pipe(
            map((event: Event) => get(event, 'data') as Req),
        ),
    );

    const response$ = merge(
        response,
        fromEvent(crossResponse, 'message').pipe(
            map((event: Event) => get(event, 'data') as Res),
        ),
    );

    return {
        emit: (data: Req) => {
            request.next(data);
            crossRequest.postMessage(data);
        },
        obs: request$,
        request: (data: Req, { withCross = true }: { withCross?: boolean } = {}) => {
            request.next(data);
            withCross && crossRequest.postMessage(data);
            return response$;
        },
        response: (cb: (data: Req) => Res | Promise<Res>) => {
            const sub = request$.subscribe(async (data) => {
                const res = await cb(data);
                response.next(res);
                crossResponse.postMessage(res);
            });

            return () => sub.unsubscribe();
        },
    };
}