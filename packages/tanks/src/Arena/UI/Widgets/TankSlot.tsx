import { TankSelector } from './TankSelector.tsx';
import { TankPreview } from './TankPreview.tsx';
import { Button } from '@heroui/react';
import { addTankToSlot, getTankEidBySlot$ } from '../../State/Game/playerMethods.ts';
import { useCallback } from 'react';
import { VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { PilotSelector } from './PilotSelector.tsx';

export function TankSlot({ className, slot }: { className: string, slot: number }) {
    const addTank = useCallback(() => addTankToSlot(VehicleType.MediumTank, slot), [slot]);
    const vehicleEid = useObservable(getTankEidBySlot$(slot));

    return (
        <div className={ `${ className } flex flex-col gap-2` }>
            { !vehicleEid && <Button onClick={ addTank }>Add tank</Button> }
            { vehicleEid && <>
                <div className="w-full grid grid-cols-2 gap-2">
                    <TankSelector className="" slot={ slot }/>
                    <PilotSelector className="" tankEid={ vehicleEid } slot={ slot } />
                </div>
                { vehicleEid && <TankPreview tankEid={ vehicleEid }/> }
            </> }
        </div>
    );
}
