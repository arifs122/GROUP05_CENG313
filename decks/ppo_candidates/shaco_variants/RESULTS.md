# Shaco → Hydrapple variants (local PPO vs PPO)

Base: `decks/ppo_candidates/mega_lopunny_yarki.csv` (the submitted `shaco`
deck).  Opponent: `decks/ppo_candidates/hydrapple_dipam.csv`, the exact
Hydrapple list seen in 20/22 of shaco's sampled ladder games.

Protocol: **20 independent arena processes × 5 games × 2 deck orders = 200
games per variant**.  Both sides run the same public PPO bundle.  This is a
worst-case structural test, not the ladder average; the local Hydrapple pilot
is stronger than the average ladder opponent.

| Variant | Changes vs base | W-L | Win rate | As P0 | As P1 |
|---|---|---:|---:|---:|---:|
| `mega_lopunny_yarki` (baseline) | — | 50-150 | **25.0%** | 23.0% | 27.0% |
| `shaco_fire_chiyu` | +1 Chi-Yu, +1 Fire; −1 Chien-Pao, −1 Battle Cage | 58-142 | **29.0%** | 34.0% | 24.0% |
| `shaco_fire_gouging` | +1 Gouging Fire ex, +2 Fire; −1 Chien-Pao, −1 Battle Cage, −1 Mist | 59-141 | **29.5%** | 23.0% | 36.0% |
| `shaco_fire_oricorio` | +1 Oricorio ex, +2 Fire; −1 Chien-Pao, −1 Battle Cage, −1 Mist | 52-148 | **26.0%** | 25.0% | 27.0% |
| `shaco_fire_judge` | +1 Chi-Yu, +1 Fire, +2 Judge; −1 Chien-Pao, −1 Battle Cage, −1 Mist, −1 Wally | 56-144 | **28.0%** | 26.0% | 30.0% |

## Reading

- All four variants still lose badly to the local PPO Hydrapple.  The best is
  Gouging Fire at 29.5%, only +4.5 pp over the 25.0% baseline; with 200 games
  the 95% interval is roughly ±6 pp, so the variants are within noise of each
  other.
- The Fire tech does not flip the matchup.  The PPO is a generalist: adding a
  one-off Fire attacker does not make it prioritise that line, and Hydrapple
  still has 2 Hydrapple ex, 4 Teal Mask Ogerpon ex, 2 Meganium, 2 Tapu Bulu,
  Lana's Aid and Night Stretcher.  One attacker cannot break that engine.
- The earlier ladder sample of 22 shaco-vs-Hydrapple episodes showed 45.5%;
  the 200-game local PPO test shows 25.0% for the same base deck.  The
  difference is opponent quality/list: the local PPO Hydrapple is a much
  stronger pilot than the average ladder Hydrapple.  Treat the local number as
  a worst-case, not the expected ladder win rate.
- Chi-Yu / Gouging Fire / Oricorio all improve the structural matchup a
  little, but none is a real answer.  A dedicated Fire shell or a different
  primary plan is needed; a 1-2 card swap is not enough.
- The clean ladder answer remains the public Alakazam 973.9 (single-prize,
  Powerful Hand one-shots 330 HP Hydrapple ex; 5-2 in the 18-episode sample).

The raw JSON results are `results/shaco_hydra_baseline.json`,
`results/shaco_hydra_shaco_fire_chiyu.json`,
`results/shaco_hydra_shaco_fire_gouging.json`,
`results/shaco_hydra_shaco_fire_oricorio.json`,
`results/shaco_hydra_shaco_fire_judge.json`.
