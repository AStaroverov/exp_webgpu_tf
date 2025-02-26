let shouldDraw = localStorage.getItem('shouldDraw') === 'true';

document.getElementById('toggleRender')!.addEventListener('click', () => {
    shouldDraw = !shouldDraw;
    localStorage.setItem('shouldDraw', shouldDraw.toString());
});

export function getDrawState(): boolean {
    return shouldDraw;
}