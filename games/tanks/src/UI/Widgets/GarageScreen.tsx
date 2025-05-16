import { TankPreview } from './TankPreview.tsx';
import { CSSProperties } from 'react';
import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';
import { GAME_MAX_TEAM_TANKS } from '../State/Game/engineMethods.ts';
import { addTank, startGame } from '../State/Game/playerMethods.ts';
import { tankEids$ } from '../State/Game/gameMethods.ts';
import { EMPTY_ARRAY } from '../../../../../lib/const.ts';

export function GarageScreen({ className, style }: {
    className?: string,
    style?: CSSProperties,
}) {
    const tankEids = useObservable(tankEids$, EMPTY_ARRAY);

    return (
        <div className={ className } style={ style }>
            <div className="flex flex-col gap-2">
                { tankEids.map((id) => {
                    return <TankPreview key={ id } className="flex grow" id={ id }/>;
                }) }
                { tankEids.length < GAME_MAX_TEAM_TANKS && <div
                    className="flex grow"
                    onClick={ addTank }
                >
                    + Add
                </div> }
                <div className="flex grow" onClick={ startGame }>
                    Start
                </div>
            </div>
        </div>
    );
}

