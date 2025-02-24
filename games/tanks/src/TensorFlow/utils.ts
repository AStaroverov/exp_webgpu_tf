let shouldDraw = false;

document.getElementById('toggleRender')!.addEventListener('click', () => {
    shouldDraw = !shouldDraw;
});

export function getDrawState() {
    return shouldDraw;
}