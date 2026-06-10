import "@tensorflow/tfjs-backend-wasm";
import { randomShortId } from "../../../../lib/random.ts";
import { setConsolePrefix } from "../../../ppo/src/infra/console.ts";
import { initTensorFlow } from "../../../ppo/src/infra/initTensorFlow.ts";
import "../../../ppo/src/infra/unhandledErrors.ts";
// Side-effect import: pulls in the network module so its custom layers and the
// AdamW optimizer self-register (tf.serialization.registerClass). The actor only
// LOADS the saved policy model — without these registrations, deserializing it
// (incl. its training config) throws "Unknown layer: AdamW".
import "../models/createUnknownNetworks.ts";
import { UnknownEpisodeManager } from "../agents/UnknownEpisodeManager.ts";

setConsolePrefix(`[ACTOR|${randomShortId()}]`);

async function initSystem() {
  const tfInitialized = await initTensorFlow("wasm");
  if (!tfInitialized) {
    console.error("Failed to initialize TensorFlow.js, aborting");
    return null;
  }

  try {
    new UnknownEpisodeManager().start();
    console.log("Actor manager initialized");
  } catch (error) {
    console.error("Failed to start", error);
    return null;
  }
}

initSystem();
