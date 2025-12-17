import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { Chip } from '../Components/Chip.tsx';
import { getVehicleState$ } from '../../State/Game/gameMethods.ts';

export function TankPreview({ className, tankEid }: { className?: string, tankEid: number }) {
    const vehicle = useObservable(getVehicleState$(tankEid));

    return (
        <div className={ `${ className } flex gap-2` }>
            <Chip color="primary">ID: { vehicle?.id }</Chip>
            <Chip color="success">Health: { vehicle?.healthAbs }</Chip>
            <Chip color="danger">Engine: { vehicle?.engine }</Chip>
        </div>
    );
}
