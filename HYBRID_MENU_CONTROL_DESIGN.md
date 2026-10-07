# Hybrid RAM-Driven Menu Control — Design Proposal

**Status:** Proposal for review — **no code written yet**
**Date:** 2026-10-07 (late evening run, fleet auto-stops 00:00)
**Author:** OpenCoder
**Scope:** Single-ROM game-beater path for Pokémon Red (Gen 1), 6-lineage PPO fleet

---

## 1. Goal

Train a Pokémon Red agent to "near game-beater" on a **single ROM**, without
hard-coding the winning route. This design replaces the low-level
"RL learns every button press" approach with a **two-layer control** where:

- **RL policy** makes the decisions (where to go, which move to use, whether to fight/flee/switch).
- **Deterministic RAM-driven menu automaton** executes the button choreography a human would press (FIGHT → move → animations → next text box).

The automaton implements the game's *mechanics*, never its *strategy* — so the
no-cheat rule still holds. Route discovery, move choice, timing, and
leveling remain learned.

---

## 2. Why the current approach plateaus (evidence from tonight)

| Observation | Detail |
|---|---|
| Step budget | Red lineage ~2.78M steps; Atari-RL "fully trained" territory |
| Result | Stuck at **Oak's Lab (map 40), Lvl 5, 0 badges** for the entire session |
| EXP fix (committed `a6c256d`) | Reward fires only *after* a battle ends; mid-battle is a ~30–60 step zero-reward tunnel with a −0.005/tick penalty |
| Behavior signature | R decaying 34→29 (red_f) over 6 rollouts at 22:33 — agents prefer fully-visited tiles over committing to battles |
| Restart cost | Each hot-restart re-boots the ROM; agent must re-walk the intro (~15 min) because checkpoints save weights, not emulator state |

**Conclusion:** The bottleneck is not exploration or reward sparsity per se — it
is that RL is forced to learn *button choreography* (menus, battle text, move
selection) as part of the same policy that should be learning *strategy*.
Pokémon Red has no published "beat the game from scratch with raw buttons"
result for exactly this reason.

Fix = separate the machine (mechanically deterministic menus) from the policy
(strategic decisions).

---

## 3. Architecture

### 3.1 Two-layer control

```
┌─────────────────────────────┐
│  RL policy (PPO, unchanged) │  high-level phase actions, e.g. {FIGHT, move2, run}
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Menu Automaton (new)       │  deterministic RAM-driven button macros
│  per environment phase      │  FIGHT→A, move2→Down+A, text→A, ...
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  PyBoy emulator (unchanged) │  existing env, reward, state extraction
└─────────────────────────────┘
```

### 3.2 Environment phases (FSM)

The environment becomes a phase state machine. Each `step(action)` runs the
automaton until the **next decision point** (or a guard budget expires), then
returns control to the policy. Detected via RAM reads (see 3.3) with a
VRAM-tilemap pattern-match fallback for robustness.

| Phase | Policy action space | Automaton executes |
|---|---|---|
| `OVERWORLD` | raw buttons (unchanged: up/down/left/right/A/B/start/select) | nothing (passthrough) |
| `TEXT` / dialogue | `{advance}` | A (dismiss text boxes, one macro per decision point) |
| `PARTY/ITEM_MENU` | `{switch1..6, use_item, close}` | menu macros |
| `BATTLE_MENU` | `{FIGHT, PKMN, ITEM, RUN}` | cursor moves + A |
| `MOVE_SELECT` | `{move1..4, back}` | directional + A on chosen move |
| `BATTLE_WAIT` (animations / enemy text / item text) | `{advance}` (or auto-advance) | A / B as needed |
| `EXP_SCREEN` / `LEVEL_UP` | `{advance}` | A through stat pages |
| `WILD_END` / `TRAINER_END` | `{advance}` | A back to overworld |

**Guard budget:** each high-level action may consume at most N emulator steps
(default 400). If the FSM hasn't reached its next decision point by then, the
step returns with `terminated=False`, phase unchanged, and the policy sees the
same menu again. This prevents a single bad abstraction from hanging a
rollout.

**`info` additions:** `phase`, `battle_type`, `enemy_hp`, `player_hp`,
`battle_wins` (lifetime), used by reward shaping and the dashboard.

### 3.3 RAM read plan

**Reuse (already validated in `orion/pokemon/environment.py`):**
party count, party level, EXP, map id, badges, in-game-world gate.

**New reads — mark 🔬 validate via probe script** (same technique as
`dbg_probe_exp.py`, which verified the EXP reward math):

| Purpose | Candidate address (pokered) | Status |
|---|---|---|
| Battle type (0=wild, 1=trainer, 2=rival…) | `$D057` (wBattleType) | 🔬 validate |
| Enemy mon HP | `$CFE0` (wEnemyMonHP) | 🔬 validate |
| Player battle mon HP | `$D00F` (wBattleMonHP) | 🔬 validate |
| Battle result / end flag | locate (candidate `$D014` region) | 🔬 validate |
| Text-box / menu-open flag | primary RAM candidate + fallback: VRAM tilemap region (`$C3A0`+ text/menu box tiles) | 🔬 validate both |

Validation scenario (scripted, headless PyBoy): boot → get starter → beat
rival → enter Viridian grass → trigger wild fight → flee → blackout recovery.
Assert each address changes at the expected moments; ship only after the
probe is green (like `PROBE OK` earlier).

### 3.4 Determinism and honesty

- The automaton only presses the same buttons a human presses for the same menu.
- It does **not** read invisible state to win: move selection uses the visible
  move list order; no enemy-team lookup, no RNG peeking, no route scripting.
- Reward caps (section 4) enforce that automation cannot fabricate reward.

---

## 4. Reward shaping spec (battle)

Keep current terms (tile explore, story progress, EXP gain `0.15/EXP`).
Add per-battle shaped terms, all validated as ≤ legitimate game maxima:

| Event | Reward | Note |
|---|---|---|
| Damage dealt to enemy (HP) | `+0.02/HP` | every battle tick where HP dropped |
| Damage to own mon | `−0.01/HP` | small; discourages suicidal trades |
| Enemy faint | `+15` | |
| Battle won (trainer) | `+40` | incl. rival/gym |
| Battle won (wild) | `+20` | |
| Capture (wild only) | `+25` | |
| Level-up | `+5` | in addition to EXP term |
| Step penalty | `−0.005` | unchanged |

**Guards:** per-battle reward cap (e.g. `max(won_bonus + 50)`); terms fire
only while `phase ∈ {BATTLE_MENU, MOVE_SELECT, BATTLE_WAIT}` for the *current*
battle; no cross-battle carry-over; dashboard will show `battle_wins`,
`battles`, `damage_total` to make farming visible.

---

## 5. Training plan (phase 2)

- **Stage A — Automaton win-rate validation:** headless scripted runs; target
  ≥90% rival-battle win over 50 runs before touching the policy.
- **Stage B — Policy resume:** resume the champion checkpoint (~2.56M steps)
  with the hybrid env; 6 lineages, same supervisor/end-time/dashboard stack,
  new `--phase` flag enables the automaton; log new fields to
  `training_log.json` (battle stats, phase histogram).
- **Success gate:** ≥2 of 6 lineages earn the **Brock badge** within 48h
  wall-clock. If not, adjust battle weights once (more aggressive damage
  term), retrain 24h, re-evaluate.
- **Roll-out:** badges 2–4 (~3–5 days), then full clear (weeks) — see §8.

**Curriculum note:** the forced intro (starter pick + scripted rival win) is
compulsory and RNG-independent — automating it as a boot script is a proposed
decision point (§8 Q1), not a "winning route".

---

## 6. Benchmark plan (the gate before writing code)

1. **Automaton micro-benchmark** — headless PyBoy, 50 runs of the scripted
   scenario (§3.3): report success %, median/max steps per phase.
2. **Throughput benchmark** — wrapped env vs current, 6 envs × 5 min:
   steps/s delta (automaton overhead must be <15% overhead; guard budget must
   not stall rollouts).
3. **Training benchmark (2h)** — 4 lineages with automaton vs 2 lineages
   baseline (EXP-only champ resume): time-to-leave-lab, time-to-first-wild-win,
   R trend. Gate: automaton lineages leave Oak's Lab and win wild battles
   strictly better than baseline.
4. **Anti-cheat/parity** — assert rewards ≤ legitimate maxima; replay recorded
   wins (live.png sequences) to confirm real battles happened.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| RAM offsets wrong / ROM revision drift | validation probes + tilemap fallback detector |
| Automaton softlock (battle stuck) | watchdog: no progress for 300 steps in a menu → bail to safest reachable state; hard cap per action (guard budget) |
| Policy spams FIGHT / overfits to automaton | reward caps + curriculum + dashboard battle-stats visibility |
| Throughput loss from guard budget | benchmark gate §6.2, tune N |
| Gen 2 freeze persists | out of scope for phase 2 (frozen; needs PyBoy-level fix later) |

---

## 8. Open questions for review (decision points)

1. **Automate the forced intro** (starter pick + scripted rival win) as a boot
   script for every lineage? *Recommendation: yes* — it is compulsory in-game
   story with no strategic choice; it removes the 15-min re-walk per restart.
2. **Battle reward weights** in §4 (damage 0.02/HP, wins 40/20, capture 25) —
   approve or adjust before implementation.
3. **Phase-2 allocation:** keep 6 Red-only lineages running continuously
   overnight (Gen 2 still frozen)? *Recommendation: yes.*
4. **Success bar:** ≥2/6 lineages beat Brock within 48h as the phase-2 gate —
   acceptable?

---

## 9. Milestones & honest timeline

| Milestone | Deliverable | Estimate |
|---|---|---|
| M0 | This proposal approved | now |
| M1 | RAM/automaton probes green + automaton μbench ≥90% rival win | ~2–4h engineering |
| M2 | Hybrid env integrated, throughput OK (§6.2) | +1–2h |
| M3 | Brock-gate training benchmark (48h fleet) | +48h wall-clock |
| M4 | Badges 2–4 | ~3–5 days |
| M5 | Full clear (8 badges + Elite Four) | ~1–3 weeks continuous |

No one-night path exists with the current architecture. This is the honest,
evidence-backed route to a single-ROM game-beater.