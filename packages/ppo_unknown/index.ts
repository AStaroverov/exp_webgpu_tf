import "./src/entry";

// Periodic hard reload to shed any accumulated GPU/WASM state during long runs.
// setInterval(
//   () => {
//     window.location.reload();
//   },
//   1000 * 60 * 60 * 1,
// );
