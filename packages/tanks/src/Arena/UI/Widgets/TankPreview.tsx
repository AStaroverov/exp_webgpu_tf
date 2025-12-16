import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { Chip } from '../Components/Chip.tsx';
import { getTankState$ } from '../../State/Game/gameMethods.ts';

export function TankPreview({ className, tankEid }: { className?: string, tankEid: number }) {
    const tank = useObservable(getTankState$(tankEid));

    return (
        <div className={ `${ className } flex gap-2` }>
            <Chip color="primary">ID: { tank?.id }</Chip>
            <Chip color="success">Health: { tank?.healthAbs }</Chip>
            <Chip color="danger">Engine: { tank?.engine }</Chip>
        </div>
    );
}
