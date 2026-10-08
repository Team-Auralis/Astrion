# Research: How to Train the Pokémon Red Agent *Faster*

**Date:** 2026-10-08
**Scope:** Multi-platform research (Exa web / arXiv / GitHub / Reddit / Bilibili / agent-reach) on faster ways to train a single-ROM Pokémon Red RL agent.
**Status:** Research only — no code changed. Proposals at the end.

---

## 1. Executive Summary (TL;DR)

Our fleet's bottleneck is **not compute and not the algorithm (PPO)**. It is **environment engineering**: the agents burn steps on menu spam, button mashing, and position loops instead of coherent exploration. Three published, code-available projects prove the same point:

1. **PokeRL (Apr 2026, Texas A&M)** — *same stack as ours* (PyBoy + PPO). Their diagnosis of why agents "degenerate into action loops, menu spam, or unproductive wandering" is a word-for-word description of our Oak's Lab plateau. Their fixes (anti-loop ×3 layers, anti-spam, map masking, dense hierarchical rewards, curriculum) took **infinite-loop episodes from 41.2% → 4.7%**, movement actions **27.2% → 68.2%**, unique tiles visited **+40.6%**, and reached **rival-battle victory ~50% at ~500k steps with only 4 envs**.
   - At our fleet's aggregate ~200 steps/s, **500k steps ≈ one 40-minute night**. This is the single biggest "faster" lever available to us.

2. **"Pokemon Red via Reinforcement Learning" (Pleines/Addis/Rubinstein/Zimmer/Preuss/Whidden, arXiv 2502.19920, Feb 2025)** — the team that **actually beat Pokémon Red** with a <10M-param PPO policy, no pretraining, starting from random buttons. Winning recipe: massive parallelism (72+ envs / PufferLib swarm), screen + visited-mask + events observations, ~25 dense reward components, LSTM policy ~5M params, **plus scripted assists for menu/fight loops** — confirming that menu control is a real bottleneck (aligns with our hybrid menu-control design).

3. **PufferAI/pokegym (2025-26)** — fork of the winning stack with published step milestones: **Badge 1 @ 9.6M steps, Badge 2 @ 35M, Badge 3 @ 404M** (72 parallel envs, γ=0.998, LSTM/BPTT h=16, visited mask). Calibrates how much *compute* a "pure RL" (no scripts) run needs; not reachable on a laptop in nights, but tells us where our ceiling is.

**Bottom line:** The fastest results won't come from a bigger fleet or a fancier algorithm. They come from fixing what the agent experiences per step (loops/spam/rewards/curriculum). PokeRL is a near-perfect template for exactly that, on our exact stack.

---

## 2. Landscape (who did what, with evidence)

| Project | Stack | Proven result | Compute to get there | Reusable for us? |
|---|---|---|---|---|
| **PokeRL** (arXiv 2604.10812, code `reddheeraj/PokemonRL`, CC BY 4.0) | PyBoy + PPO (same as ours) | House-exit 65% @150k steps, tall-grass 60% @500k, **rival battle win ~50% @500k**; loops 41.2%→4.7% | 4 parallel envs (CPU+GPU) | ✅ **Near-perfect fit** — env-side fixes only |
| **PokemonRedExperiments** (PWhiddy, 2023) | PyBoy + Gym + PPO | Viral replay; V2 reached Cerulean w/ coordinate-exploration reward | Large (multi-day runs, many envs) | ⚠️ Design lessons (visited mask, level reward) |
| **"Pokemon Red via RL"** (arXiv 2502.19920 + drubinstein.github.io/pokerl, 2025) | PyBoy, PufferLib swarm, PPO + **scripts** | **Beat all 8 badges / beat Red** (Feb 2025), <10M params | 72+ envs on PufferTank-class hardware | ✅ Recipe + infra ideas; scripts conflict w/ our "no hard route" rule |
| **PufferAI/pokegym** (2025-26) | PyBoy, PufferLib, recurrent PPO (BPTT h=16, γ=0.998) | Badge 1 @ 9.6M, Badge 2 @ 35M, Badge 3 @ 404M steps | 72 parallel envs, weeks | ✅ Milestone-calibration numbers |
| **IndigoRL** (OutFerz/indigoRL, 2026) | PyBoy + SB3 Recurrent PPO (LSTM) | Progress through Viridian Forest (Yellow); skips Oak's intro via savestate (**~20% compute saving**) | 4-8 envs, GPU | ✅ Savestate-intro-skip + LSTM + RAM-feature fusion w/ CNN |
| **PokeDreamer** (xoTEMPESTox, 2025-26) | PyBoy, Dreamer-style world model | Map-ID probe 98.7% from latent; **0 badges** — experimental | Research | ⚠️ Future; not proven for progression |
| **Go-Explore** (Ecoffet et al. 2021) | N/A (algorithm) | 43k pts Montezuma w/o domain knowledge; 18M with | N/A | ⚠️ Research-stage for Pokémon (PokeRL cites as future work) |
| **IBRL / single-demo curricula** (arXiv 1812.03381 et al.) | N/A (algorithm) | Exponential→quadratic sparse-reward exploration | N/A | ⚠️ Promising curriculum idea, unproven on Pokémon |
| **metamon** (arXiv 2504.04395) | Offline RL | Competitive battles only (not story) | Large | ❌ Wrong domain |

---

## 3. The Core Diagnosis: Why We're Stuck at Level 5 / Oak's Lab

PokeRL's stated motivation matches our fleet report exactly:

> "Agents often degenerate into action loops, menu spam, or unproductive wandering … training remains brittle in practice."

Why our ~16.9M fleet steps (6 lineages × ~2.8M) bought almost nothing:

1. **Unstructured action space** — Start/Select/menu spam wastes steps and produces no state change; the policy never learns that pressing B during a text box advances the game because the reward is too sparse.
2. **No spatial memory** — without a visited/coverage signal, exploration is directionless; the agent vibes between two tiles (position loop) indefinitely.
3. **Reward too sparse and too late** — EXP-gain alone doesn't fire until battles; there is no dense progress gradient (tiles, doors, map transitions, grass) so nothing reinforces the *behaviors* that lead to battles.
4. **No curriculum** — full-game horizon from the forced intro onward; PokeRL showed the task splits (leave house → explore town → reach grass → win rival battle) are learnable individually at 150-500k steps, but nearly impossible as one long-horizon problem.

Our existing hybrid-menu-control design doc already anticipates the menu problem; the community consensus (both PokeRL and the 2025 beating team) confirms menus are a genuine bottleneck and a scripted layer is legitimate — the beating team itself used scripted fight/menu assists.

---

## 4. The Near-Perfect Win: Copy PokeRL's Env Design (not their whole repo, but their ideas)

PokeRL is CC BY 4.0, PyBoy-based, and its ideas port directly into `orion/pokemon/environment.py`. The four changes with the biggest speedup per engineering hour:

### 4.1 Three-layer anti-loop system
- **Position visit penalties:** track per-episode `(x, y)` counts; small negative reward past 3 visits on a tile, larger past 5.
- **Position loop detection:** if the agent keeps returning within a small radius of a recent location, penalize as a loop.
- **Action-pattern detection:** repeated short action sequences with no state change → penalize / mask.
- **Result:** loop episodes 41.2% → 4.7% (their measured numbers).

### 4.2 Anti-spam + input hygiene
- Remove Start/Select from the action space (adds nothing for progression; pure spam).
- Double-press handling for movement (one press+release per move; prevents toggle jitter).
- Graduated streak penalties + small diversity bonus for varying actions.
- **Result:** movement actions 27.2% → 68.2% of all actions; action-distribution entropy 1.21 → 1.82 bits.

### 4.3 Dense hierarchical rewards (micro / meso / macro)
Their published reward table (adapt weights to our balance):
- **Micro:** new tile +1.0 · distance-to-goal +0.2 · first-ever visit to tile +0.5
- **Meso:** first entry into a new map +5 · any map transition +10 · entering an unexplored region +2 · **tall grass +20** · starting a battle +10
- **Macro:** catch/win battle +50 (keep our EXP-gain as a sub-signal, but status flags matter more)
- **Result:** Pallet Town coverage 12% → 41%; unique tiles/episode +40.6%.

### 4.4 Curriculum over early-game sequences
- **Seq 1: House exit** (start in bedroom → out the door).
- **Seq 2: Pallet exploration** (reach tall grass at town edge).
- **Seq 3: Rival battle** (win first battle).
- Implement via savestates (PyBoy supports state save/load) — same trick IndigoRL used to skip Oak's intro for **~20% compute saving**.
- This also answers design-doc §8 Q1 (automate forced intro): **yes — load a savestate after Oak's intro; don't pay 20% of the episode budget re-watching text.**

### 4.5 Observation: add a visited mask (or coverage features from RAM)
We already extract RAM features; the winning stacks all add spatial memory:
- Cheapest: keep RAM features and add per-map `visited` bitmask/coverage counters (we already track map + coords in `knowledge.py` / `_extract_state_features`).
- Strongest (pokegym/pokerl): 72×80 downsampled screen + visited mask + events array — but that changes the policy input and needs a CNN; **defer until M2** if the RAM+mask version shows progress.

### 4.6 Policy: add recurrence + long γ
- Recurrent PPO (LSTM) + BPTT horizon ~16, γ ≈ 0.998 (both winning stacks use this); long horizon matters for a game where rewards fire minutes apart.
- Our PPO trainer already exists; add LSTM head + horizon + γ change (small diff).

---

## 5. Honest Compute Calibration (what's realistic on this laptop)

| Milestone | Steps needed (reported) | At our ~200 steps/s aggregate |
|---|---|---|
| House exit / grass / rival win (PokeRL curriculum) | ~500k | **~40 min** ✅ |
| Badge 1 (pokegym, pure RL) | 9.6M | ~13 h (≈ 4-5 × 3h nights) ⚠️ |
| Badge 2 (pokegym, pure RL) | 35M | ~48 h total ⚠️ |
| Badge 3 (pokegym, pure RL) | 404M | ~23 days of 24/7 ❌ laptop |

Implications:
- **The PokeRL-style env fix should get us out of the plateau in the first night** — that's the "faster results" the user wants.
- Badge 1 in ~2-4 nights *is* plausible **if** we also adopt the visited-mask + curriculum + anti-loop package. Badge 1 with our current env (symptom-heavy, no anti-loop) is not plausible at any nightly budget.
- Badges 2-3 on a laptop in nights: not with pure RL at reported scales. Options later: scripted assists (like the beating team), hybrid menu-control layer (already designed), or much more parallel CPU envs (cheap PyBoy instances are CPU-bound; 12-24 envs × 30 steps/s would materially cut wall-clock — RAM is the limiting factor at 8 GB).

---

## 6. Research-Stage Ideas — Do NOT Invest Yet

| Idea | Why not now |
|---|---|
| World models (DreamerV3 / PokeDreamer) | Best-in-class sample efficiency in theory; **0 badges on Pokémon** in practice; big engineering lift. Revisit after Badge 1. |
| Go-Explore / single-demo curricula | Proven on Montezuma but **unproven on Pokémon**; needs savestate-archive machinery we can build later on top of the PokeRL env. |
| Offline RL / BC pretraining | The beating team tried BC "once and struggled with data collection"; our bottleneck is env quality, not data quality. |
| LLM-guided play (Claude/Pimanrules) | Not RL; out of scope for "train the gaming model". |

---

## 7. Recommended Next Steps (if user approves)

**M1 (one night of engineering, ~2-4 h):** PokeRL-style env fixes in `orion/pokemon/environment.py` + `train_pokemon.py`:
1. Anti-loop×3 + anti-spam + remove Start/Select + double-press movement
2. Dense hierarchical rewards (tiles/grass/battle/events) with EXP as sub-signal
3. Per-map visited-mask observation added to `_extract_state_features`
4. Oak's-intro savestate skip (answers design-doc §8 Q1: yes)
5. Recurrent PPO (LSTM head, BPTT h≈16, γ≈0.998)
6. Curriculum: house-exit → Pallet → rival-battle savestate start points

**M2 (only if M1 shows the plateau breaking):** 72×80 screen + visited mask observation with CNN; push env count 6 → 12-24 (CPU-bound PyBoy; watch RAM).

**Then:** 2-night A/B vs. current batch to prove the fix before the 48h Brock run.

---

## 8. Sources

- PokeRL paper (arXiv 2604.10812), code: github.com/reddheeraj/PokemonRL — **primary template**
- "Pokemon Red via Reinforcement Learning" (arXiv 2502.19920) + full recipe: drubinstein.github.io/pokerl (Observations/Rewards/Policy/Swarm chapters)
- PufferAI/pokegym README (badge step milestones, γ, BPTT, visited mask)
- PWhiddy/PokemonRedExperiments (origin; coordinate-exploration reward)
- OutFerz/indigoRL (savestate intro skip ~20%, Recurrent PPO + PyBoy + SB3)
- xoTEMPESTox/PokeDreamer (world-model status)
- Ecoffet et al., Go-Explore (2021); "Exploration from a Single Demonstration" (arXiv 1812.03381)
- NeurIPS 2025 PokeAgent Challenge (competitive/long-context Pokémon at scale)

---

## 9. Answers this gives to the open design-doc §8 questions

- **Q1 Automate forced intro?** → **Yes.** Savestate-skip Oak's intro (IndigoRL: ~20% compute saving). Don't burn 20% of budget on text.
- **Q2 Battle-reward weights?** → Keep EXP but make status events dominant: start battle +10, win +50, tall-grass +20, map transition +10 (PokeRL table).
- **Q3 Nightly 6-lineage allocation?** → Keep 6 lineages but make them *curriculum-diverse* (2× house-exit, 2× Pallet, 2× rival) instead of 6 identical full-game runs; or 6× PokeRL-style envs. Revisit after M1.
- **Q4 Brock 48h gate ≥2/6?** → With env fixes, expect the gate to be *house-exit/grass/rival* milestones, not badges. A more realistic gate: ≥3/6 lineages complete the rival-battle curriculum within 48h; then promote the best lineage to full-game.