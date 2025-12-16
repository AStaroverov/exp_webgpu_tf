import { CSSProperties } from 'react';
import { startGame } from '../../State/Game/playerMethods.ts';
import { Button } from '../Components/Button.tsx';
import { Card } from '../Components/Card.tsx';
import { TankSlot } from './TankSlot.tsx';
import { range } from 'lodash-es';
import { GAME_MAX_TEAM_TANKS } from '../../State/Game/engineMethods.ts';

export function GarageScreen({ className, style }: {
    className?: string,
    style?: CSSProperties,
}) {
    return (
        <div className={ `${ className } backdrop-blur-2xl bg-amber-100 p-3 rounded-md` } style={ style }>
            <div className="flex flex-col gap-2">
                { range(0, GAME_MAX_TEAM_TANKS).map((i) => {
                    return <Card key={ i } className="p-2">
                        <div>Tank { i + 1 }</div>
                        <TankSlot className="flex grow" slot={ i }/>
                    </Card>;
                }) }
                <Button
                    className="flex grow"
                    color="primary"
                    onPress={ startGame }
                >
                    Start
                </Button>
            </div>
        </div>
    );
}

