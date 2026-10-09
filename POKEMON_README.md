# ORION Pokemon AI

Adapts ORION's ResNet architecture for Pokemon Gen 1 (Red/Blue/Yellow) gameplay, with Gen 2 support planned.

## Architecture

```
Screen (160x144) ──► Downsample 2x ──► 80x72 ──► Frame Stack (4) ──► CNN Branch ──► 256-dim
                                                                                          ├──► Fusion ──► ResNet Blocks ──► Policy (8 buttons)
Game State (RAM) ──► 20 features ──► MLP Branch ──► 64-dim                              ──► Value (expected return)
```

- **Screen branch**: Conv layers + ResNet blocks process visual game state
- **State branch**: MLP processes Pokemon-specific RAM features (badges, party, levels, etc.)
- **Dual heads**: Policy (which button to press) + Value (expected cumulative reward)

## Quick Start

### 1. Install Dependencies

```bash
pip install pyboy gymnasium torch numpy
```

### 2. Place Your ROM

Put your Pokemon ROM at:
```
roms/pokemon_red.gb    (or .gbc for Gen 2)
```

> **Note**: We don't distribute ROMs. Use your own legally obtained copy.

### 3. Test Pipeline (No ROM Required)

```bash
# Verify network, encoder, and PPO training work
python test_pokemon_pipeline.py
python test_ppo_training.py
```

### 4. Train

```bash
# Basic training
python train_pokemon.py --rom roms/pokemon_red.gb --episodes 1000

# With custom settings
python train_pokemon.py \
    --rom roms/pokemon_red.gb \
    --episodes 5000 \
    --rollout-length 4096 \
    --batch-size 128 \
    --lr 2.5e-4 \
    --entropy-coef 0.01 \
    --checkpoint-dir checkpoints/pokemon/red_v1

# Resume from checkpoint
python train_pokemon.py \
    --rom roms/pokemon_red.gb \
    --resume checkpoints/pokemon/red_v1/ppo_latest.pt
```

### 5. Evaluate

```bash
python evaluate_pokemon.py \
    --rom roms/pokemon_red.gb \
    --checkpoint checkpoints/pokemon/red_v1/ppo_best.pt \
    --episodes 20
```

## M1 Training Knobs (PokeRL env + curriculum + recurrent PPO)

The M1 bundle (from `RESEARCH_FASTER_TRAINING.md` §7) adds opt-in training
knobs. **All of them default OFF, so the supervisor fleet and existing runs are
unchanged / byte-identical.** Turning any of them on is a *new run* that
changes the action space, the state-vector size and/or the model, so
**checkpoints are not portable across modes**: an old checkpoint will not load
under `--pokerl` or `--recurrent`.

| Flag | Effect | Default |
|------|--------|---------|
| `--pokerl` | PokeRL input hygiene: 6-button action space `a,b,up,down,left,right` (no start/select) and tapped movement (one tile per decision). State vector grows 20 → 30 (adds coverage slots). | off (8 buttons, 20-dim) |
| `--curriculum {bedroom,pallet,rival}` | Resolve `states/red_<stage>.state` into `--state` to skip the forced intro (mutually exclusive with `--state`). |
| `--recurrent` | Use `PokemonRecurrentNet` (LSTM policy, truncated BPTT). |
| `--bptt-horizon` | BPTT segment length for recurrent PPO. | `16` |
| `--gamma` | Discount factor. Defaults to **0.998 with `--recurrent`**, else `0.99`; an explicit `--gamma` always wins. |

**Defaults (all flags off):** the supervisor fleet and every existing run are
unaffected and byte-identical — the same 8-button action space, 20-dim state,
and `PokemonResNet` policy as before.

> **Compatibility:** `--pokerl` changes the action space (8 → 6) and state dim
> (20 → 30); `--recurrent` changes the model (adds an LSTM). Old checkpoints do
> **not** load under either flag. Start fresh runs with new checkpoint dirs.

### Prerequisites: curriculum savestates

The curriculum starts are recorded once (they are gitignored, ~164 KB each):

```bash
python record_intro_state.py roms/pokemon_red.gb   # writes states/red_bedroom.state + states/red_pallet.state
```

`bedroom` and `pallet` are auto-recorded; `rival` must be saved manually (play
to the in-lab rival encounter and place it at `states/red_rival.state`).

### Recommended M1 launch

```bash
# Full M1 (curriculum + PokeRL env + recurrent PPO):
python train_pokemon.py --rom roms/pokemon_red.gb \
    --pokerl --curriculum pallet --recurrent --bptt-horizon 16 --gamma 0.998

# Fallback (if LSTM overhead is too high on the laptop): drop --recurrent,
# keeping the PokeRL env + curriculum fast path:
python train_pokemon.py --rom roms/pokemon_red.gb \
    --pokerl --curriculum pallet
```

### What M1 adds (and where it is probed)

- **Anti-loop + anti-spam** — three position/pattern layers, streak penalties,
  a small diversity bonus, all `--pokerl`-only (`orion/pokemon/environment.py`).
- **Dense hierarchical rewards** — micro (tile/first-visit/distance), meso
  (map entry/transition/region/tall grass/battle start), macro (battle
  win/catch); EXP stays a sub-signal.
- **Per-map coverage observation** — the state vector grows 20 → 30 under
  `--pokerl` (current + recent-map coverage slots).
- **Savestate curriculum** — `--curriculum` boots past the forced intro from
  `states/*.state` (`record_intro_state.py`).
- **Recurrent PPO** — `PokemonRecurrentNet` LSTM head with `--bptt-horizon` and
  the γ≈0.998 default.

Probes (each prints `PROBE OK` and exits 0):

```
python dbg_probe_m1.py          # consolidated M1 bundle (this section)
python dbg_probe_antiloop.py    # anti-loop / anti-spam magnitudes
python dbg_probe_pokerl.py      # hierarchical rewards
python dbg_probe_coverage.py    # coverage observation / state_dim 30
python dbg_probe_exp.py         # EXP sub-signal
```

## Reward Shaping

The environment provides milestone-based rewards:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Badge earned | +1000 | Major progression |
| New Pokemon caught | +500 | Team building |
| Pokemon leveled up | +50/level | Grinding progress |
| New map entered | +100 | Exploration |
| HP healed | +200 | Recovery |
| Talk to NPC | +10 | Information gathering |
| In battle | -1/step | Encourages efficiency |
| Step penalty | -0.1 | Encourages progress |
| Game complete (8 badges) | +5000 | Final goal |

## Gen 1 RAM State Features

The encoder extracts 20 features from emulator RAM:

```
[0]  Badges (0-8, normalized)
[1]  Party count (0-6)
[2]  First Pokemon HP (0-1)
[3]  First Pokemon level (0-100)
[4]  Map ID (0-255)
[5]  In battle flag
[6]  Player X position
[7]  Player Y position
[8]  Money (0-999999)
[9]  Experience points
[10] Pokemon 1 level
[11] Pokemon 2 level
[12] Pokemon 3 level
[13] Pokemon 1 HP
[14] Pokemon 2 HP
[15] Pokemon 3 HP
[16] Game state
[17] Dialogue active
[18] Steps taken (episode progress)
[19] Maps visited (exploration progress)
```

## File Structure

```
orion/pokemon/
├── __init__.py          # Package exports
├── environment.py       # PyBoy Gymnasium wrapper (Gen 1)
├── network.py           # Dual-branch ResNet + Encoder
├── mock_env.py          # Synthetic environment for testing

train_pokemon.py         # PPO training script
evaluate_pokemon.py      # Evaluation script
test_pokemon_pipeline.py # Unit tests (network, encoder, PPO)
test_ppo_training.py     # End-to-end training test
```

## Training Tips

### CPU vs GPU

The training auto-detects CUDA. On CPU, expect:
- ~50-100 steps/s (rollout collection)
- ~200-500 steps/s (PPO updates)

On GPU:
- ~200-500 steps/s (rollout)
- ~2000-5000 steps/s (updates)

### Checkpointing

Checkpoints save every 10 rollouts:
- `ppo_latest.pt` - Most recent (for resuming)
- `ppo_best.pt` - Best episode reward
- `ppo_final.pt` - Final state
- `training_log.json` - Metrics history

### Hyperparameter Guidelines

| Parameter | Default | When to Adjust |
|-----------|---------|----------------|
| `--lr` | 3e-4 | Lower (1e-4) if unstable |
| `--entropy-coef` | 0.02 | Higher (0.05) if converging too early |
| `--rollout-length` | 2048 | Higher (4096) for better advantage estimates |
| `--ppo-epochs` | 4 | Lower (2) if overfitting |
| `--gamma` | 0.99 | Higher (0.995) for longer horizons |

## Gen 2 Support (Planned)

Gen 2 (Gold/Silver/Crystal) requires:
1. Different RAM addresses (mostly compatible but offsets shift)
2. 2-byte extended features (day/night, pokegear)
3. Crystal has animated battle sprites

The environment structure is ready - we'll add `Gen2RAM` class and a `PokemonGen2Env` subclass.

## Troubleshooting

### "ROM not found"
Place your ROM at the specified path. The emulator needs a valid `.gb` or `.gbc` file.

### "CUDA out of memory"
Reduce `--batch-size` (try 64) or `--rollout-length` (try 1024).

### Training not progressing
1. Check `training_log.json` - is reward increasing?
2. Increase `--entropy-coef` to 0.05 for more exploration
3. Verify ROM is correct game (Red vs Blue have different RAM)

### Emulator window doesn't close
Use `--headless` (default) for training. Only use `--render` for watching.

## How It Works

1. **Observation**: Environment captures 160x144 screen + reads 20 RAM addresses
2. **Encoding**: Screen downsampled to 80x72, frame-stacked (4 frames for motion)
3. **Forward pass**: CNN processes screen, MLP processes state, fusion combines them
4. **Action**: Policy outputs probability over 8 buttons (A, B, Start, Select, D-pad)
5. **Reward**: State changes (badges, levels, maps) generate shaped rewards
6. **PPO update**: GAE advantages + clipped policy gradient + value function learning

## Credits

- **PyBoy**: Game Boy emulator - https://github.com/Baekalfen/PyBoy
- **ORION**: Chess engine that inspired this architecture
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
