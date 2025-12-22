import { TankPreview } from './TankPreview.tsx';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { playerVehicleEid$ } from '../../State/Game/modules/player.ts';

export function PlayerSlot({ className }: { className?: string }) {
    const vehicleEid = useObservable(playerVehicleEid$);

    return (
        <div className={ `${ className } flex flex-col gap-2` }>
            <div className="text-sm font-semibold text-amber-800">Harvester</div>
            { vehicleEid && <TankPreview tankEid={ vehicleEid }/> }
        </div>
    );
}

