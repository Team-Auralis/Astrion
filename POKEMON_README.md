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
