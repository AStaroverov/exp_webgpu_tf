import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';
import { getTankState$ } from '../State/Game/gameMethods.ts';

export function TankPreview({ className, id }: { className: string, id: number }) {
    const tank = useObservable(getTankState$(id));

    return (
        <div className={ className }>
            { tank?.id } / { tank?.health }
        </div>
    );
}
