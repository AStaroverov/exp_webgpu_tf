import devtoolsDetect from 'devtools-detect';

let isVerbose = localStorage.getItem('verbose') === 'true';
let shouldDraw = localStorage.getItem('shouldDraw') === 'true';

document.getElementById('toggleRender')!.addEventListener('click', () => {
    shouldDraw = !shouldDraw;
    localStorage.setItem('shouldDraw', shouldDraw.toString());
});

document.getElementById('toggleVerbose')!.addEventListener('click', () => {
    isVerbose = !isVerbose;
    localStorage.setItem('verbose', isVerbose.toString());
});

export function getDrawState(): boolean {
    return shouldDraw;
}

export function isVerboseLog() {
    return isVerbose && devtoolsDetect.isOpen;
}