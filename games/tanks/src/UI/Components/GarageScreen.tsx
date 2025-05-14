import { TankPreview } from './TankPreview.tsx';
import { CSSProperties } from 'react';
import { GameTankState } from '../Hooks/useGameAPI.ts';

export function GarageScreen({ className, style, tanks, handleAddTank, handleStart }: {
    className?: string,
    style?: CSSProperties,
    tanks: GameTankState[],
    handleAddTank?: () => void,
    handleStart?: () => void
}) {
    return (
        <div className={ className } style={ style }>
            <div className="flex flex-col gap-2">
                { tanks.map((t) => {
                    return <TankPreview key={ t.id } className="flex grow" tank={ t }/>;
                }) }
                { handleAddTank && <div
                    className="flex grow"
                    onClick={ handleAddTank }
                >
                    + Add
                </div> }
                <div
                    className="flex grow"
                    onClick={ handleStart }
                >
                    Start
                </div>
            </div>
        </div>
    );
}

