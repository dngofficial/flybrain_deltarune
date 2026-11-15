# Flybrain controller for EnemyRecreations Rudinn

<img width="1771" height="1159" alt="image" src="https://github.com/user-attachments/assets/ccc8bb61-30fe-4075-9b3a-009faab53071" />


This is the existing flybrain bridge configured for the original Rudinn fight in the [EnemyRecreations project](https://github.com/KristalTeam/EnemyRecreations). It includes no new encounter, wave, or bullet and no Python game simulation. Kristal sends its rendered screenshot to Python during the Rudinn defense phase; Python returns a movement action. Confirmed bullet-hit counts and episode state are sent as metadata for learning.

## Install

1. Install Kristal and the EnemyRecreations mod, including its existing `libraries/Rudinn` folder.
2. Copy `kristal/flybrain` from this package into the mod's `libraries` folder. You should have both `EnemyRecreations/libraries/Rudinn/` and `EnemyRecreations/libraries/flybrain/`. The bridge declares a dependency on the Rudinn library.
3. From this extracted package, install its existing Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

On Linux/macOS, create a venv with `python3 -m venv .venv`, activate it and run `python -m pip install -e .`.

4. Start the controller:

```powershell
.\.venv\Scripts\python.exe -m flybrain serve --mode reflex
```

5. Open EnemyRecreations, select its **Rudinn** encounter, and press **F8** in Kristal. Play the battle menus yourself; the controller takes over only while a red soul can move during a Rudinn defense wave. Press F8 again for manual control. The game uses the original Rudinn enemy, waves and bullets.

The inspector shows the exact latest screenshot processed, the occupancy grid derived from it, the connectome activity, action and reward. It clears the frame when a wave ends or the connection closes. It does not show a Python recreation of the game.

## Train on real Rudinn frames

```powershell
.\.venv\Scripts\python.exe -m flybrain serve --mode policy --learn --out checkpoints/rudinn.pt
```

This starts an **untrained** policy, so it will need training and may take damage. The reward is `0.02 × simulated seconds − 1 per confirmed hit − 2 on death`. Kristal controls the real battle and its health rules. Repeat Rudinn waves through the normal menus; the controller does not automatically restart battles or navigate menus. Checkpoints are saved after PPO updates and at exit. To resume:

```powershell
.\.venv\Scripts\python.exe -m flybrain serve --mode policy --learn --checkpoint checkpoints/rudinn.pt --out checkpoints/rudinn.pt
```

To use the saved policy without training, omit `--learn`. A reflex baseline is available for checking the connection; it is not a learned fly policy.

The included MaleCNS subset contains 1,024 real neurons and 62,888 directed connections. The rate dynamics, pixel-to-neuron adapter, and trainable motor readout are artificial; only the readout is trained. The full 125-million-synapse nervous system is not simulated. Source: [MaleCNS v1.0](https://male-cns.janelia.org/download/), FlyEM/HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included subset was selected and thresholded from the source data.

## Notes

- The controller expects a green rectangular arena, a red soul, and visible white or white-outlined black bullets. Pure black on black cannot be seen. Motion is estimated from consecutive screenshots.
- The library uses fixed-step lockstep: game simulation waits for Python's response. A disconnect or 15-second timeout returns control to the player. The server listens on `127.0.0.1:65432`.
- It was originally developed against Kristal 0.11.0-dev. EnemyRecreations advertises an older engine version; compatibility with your installed version depends on Kristal APIs and has not been verified for every release.
- The Python inspector requires a desktop. Use `--no-ui` to run without the window. The project does not control the retail DELTARUNE game.
