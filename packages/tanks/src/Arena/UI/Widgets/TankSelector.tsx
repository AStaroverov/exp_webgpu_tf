import { Select, SelectItem } from '../Components/Selector.tsx';
import { VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { changeTankType, getVehicleType$ } from '../../State/Game/gameMethods.ts';
import { EMPTY } from 'rxjs';
import { ChangeEvent, useCallback } from 'react';
import { getSlot$ } from '../../State/Game/modules/lobbySlots.ts';

const vehicleTypes = [
    { key: VehicleType.LightTank, label: 'Light' },
    { key: VehicleType.MediumTank, label: 'Medium' },
    { key: VehicleType.HeavyTank, label: 'Heavy' },
];

export function TankSelector({ className, slot }: { className?: string, slot: number, }) {
    const vehicleEid = useObservable(getSlot$(slot));
    const vehicleType = useObservable(vehicleEid ? getVehicleType$(vehicleEid) : EMPTY);
    const handleChangeVehicleType = useCallback((event: ChangeEvent<{ value: string }>) => {
        changeTankType(vehicleEid, slot, Number(event.target.value));
    }, [vehicleEid]);

    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Tank type"
                value={ vehicleType }
                onChange={ handleChangeVehicleType }
            >
                { vehicleTypes.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}
