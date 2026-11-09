# Hydrapple item variants vs Alakazam 1074 — 200 BO3

Opponent: `decks/ppo_candidates/alakazam_1074.csv` (the public f23b6ed606
Alakazam).  Both sides: public PPO bundle.  Protocol: **20 independent arena
processes × 5 BO3 matches × 2 deck orders = 200 BO3 matches per variant**
(100 as P0, 100 as P1).

## Results

| Variant | Changes vs `hydrapple_dipam` | Hydra W-L | Hydra BO3 WR | P0 | P1 |
|---|---|---:|---:|---:|---:|
| **baseline** | — | 33-167 | **16.5%** | 16.0% | 17.0% |
| `hydra_herocape` | +Hero's Cape, −Unfair Stamp | 2-198 | **1.0%** | 1.0% | 1.0% |
| `hydra_survival_heal` | +Survival Brace, +Jumbo Ice Cream, −Unfair Stamp, −1 Dawn | 4-196 | **2.0%** | 1.0% | 3.0% |
| `hydra_payapa_sacred` | +2 Payapa Berry, +2 Sacred Charm; −1 Boss, −1 Dawn, −1 Poké Pad, −1 Tapu Bulu | 16-184 | **8.0%** | 4.0% | 12.0% |
| `hydra_judge_payapa` | +2 Judge, +2 Payapa Berry; −1 Boss, −1 Dawn, −1 Poké Pad, −1 Tapu Bulu | 20-180 | **10.0%** | 10.0% | 10.0% |

Every item variant is **worse** than the original Hydrapple.  The best
Hydrapple configuration against Alakazam is still the baseline.

## Why the HP item did not work

Alakazam's `Powerful Hand` does 20 damage per card in hand.  In the sampled
ladder Hydrapple games:

- 11-14 cards → **220-280** damage.  Hydrapple ex (330 HP) already survives
  this; it does not need +100 HP for the common case.
- 17-19 cards → **340-380** damage.  This is the one-shot range; this is where
  Hero's Cape (+100 HP → 430) or Payapa Berry (−60 per Psychic hit) would
  matter.
- The base Hydrapple already has **1 Judge + 1 Unfair Stamp** to attack that
  hand size.  Replacing Unfair Stamp with Hero's Cape removes the hand
  disruption that prevents the 17-19 card turns; the 1.0% result shows the
  trade is catastrophic.
- `Hero's Cape` and `Survival Brace` are **ACE SPEC**, so they cannot coexist
  with Unfair Stamp.  The ACE slot is worth more as Unfair Stamp than as
  +100 HP.
- The PPO is a generalist: it does not prioritise attaching a one-off tool to
  Hydrapple ex.  The tool often sits in hand or goes to the wrong Pokémon.
- The non-ACE variants (Payapa/Sacred, Judge/Payapa) kept Unfair Stamp but cut
  Boss/Dawn/Poké Pad/Tapu Bulu to make room; the consistency loss outweighed
  the damage reduction.  They are 8-10% vs the 16.5% baseline.

## Conclusion

- **HP/damage-reduction items do not solve Hydrapple vs Alakazam** with this
  PPO.  The ACE SPEC cost (losing Unfair Stamp) and the PPO's inability to
  prioritise situational tools make them strictly worse.
- Alakazam's advantage is structural: single-prize attacker, scaling damage,
  and a deck that keeps coming back; Hydrapple's ex-heavy board and prize map
  lose the long game.  Alakazam 1074 is the clean ladder answer (973.9 public
  score, 75% local vs Hydrapple, 5-2 in the ladder sample).
- If this line is pursued further, the only sensible test is a **minimal
  non-ACE item** (e.g. +1 Payapa Berry, −1 Tapu Bulu) with a rule-based
  "attach to Hydrapple ex" override, not a multi-card rebuild.  The present
  data says the expected gain is small.

Raw results: `results/hydra_alakazam_bo3_*.json`.

## Unfair Stamp hold test (200 BO3)

Unfair Stamp is **ACE SPEC**, so a deck may contain only one copy.  "Increasing"
it is illegal.  The legal way to add hand disruption is Judge; the timing idea
was implemented as a rule in a custom PPO bundle: when Unfair Stamp is in hand
and the opponent's public `handCount` is below 17, the Unfair option is masked
out of the model's action list and the model must choose another legal action.
At 17+ the option is left available.

| Run | Deck / rule | Hydra W-L | Hydra BO3 WR |
|---|---|---:|---:|
| baseline (standard PPO) | `hydrapple_dipam` | 33-167 | **16.5%** |
| baseline + hold≥17 | `hydrapple_dipam`, Unfair held until opp hand ≥17 | 12-188 | **6.0%** |
| Judge3 (standard PPO) | +2 Judge, −1 Dawn, −1 Poké Pad | 19-181 | **9.5%** |
| Judge3 + hold≥17 | same deck, Unfair held until opp hand ≥17 | 20-180 | **10.0%** |

All Unfair/Judge variants are worse than the 16.5% baseline.

Why:

- In the sampled Alakazam–Hydrapple games, Alakazam's hand at the Unfair-legal
  windows was usually **9-14**, with rare 15-16 and one 19.  Holding until 17+
  makes Unfair mostly a dead card; using it at 11-14 still disrupts Alakazam's
  next draw and is better than skipping the turn's action.
- The hold implementation also forces the PPO to take a different legal action
  whenever the stamp is masked, which is often worse than the policy's choice.
- Judge3 required cutting Dawn/Poké Pad; the consistency loss outweighed the
  extra disruption.  Extra Judge does not compensate for the structural
  Hydrapple disadvantage.
- Unfair Stamp's ACE SPEC slot cannot be duplicated; the base deck already has
  the best legal package (1 Unfair + 1 Judge).

The best Hydrapple configuration against Alakazam remains the original
`hydrapple_dipam` at 16.5% BO3; the matchup is structural, not a timing issue.

## Smart Judge / Unfair timing rule (200 BO3)

Rule implemented in a custom PPO `main.py`:

- opponent hand **9-10** → force Judge (if legal);
- opponent hand **17+** → force Unfair Stamp (the "jump" from 9);
- if no 17+ jump ever happened and the opponent is back at 9-10 late
  (turn ≥ 8 or a player at ≤2 prizes) → force Unfair Stamp at 9-10;
- otherwise mask Judge and Unfair out of the model's options so they are held.

The rule fired correctly in a smoke test (2 Judge + 1 Unfair in two BO3), but
the results are disastrous:

| Run | Hydra W-L | Hydra BO3 WR |
|---|---:|---:|
| baseline, standard PPO | 33-167 | **16.5%** |
| baseline, smart rule | 3-197 | **1.5%** |
| Judge3, standard PPO | 19-181 | **9.5%** |
| Judge3, smart rule | 4-196 | **2.0%** |

Why the hand-coded timing loses:

- Forcing Judge at 9-10 spends the Supporter for the turn; Hydrapple loses the
  attack/setup tempo.  Alakazam simply rebuilds.
- Masking Judge/Unfair at 11-16 removes plays the PPO would otherwise use well;
  the policy's own timing is better than the 9-10 heuristic.
- The 3-Judge deck already pays a consistency cost (Dawn/Poké Pad cuts); forced
  supporters make it worse.
- The opponent's hand can jump from counts other than 9-10; the simple rule
  misses those windows, while the PPO's policy sees the full board.

Conclusion: do not override the PPO with this hand-count rule.  The baseline
16.5% is still the best Hydrapple result; the matchup is structural and the
policy's own Unfair/Judge usage is better than the hand-coded timing.

## Speed / no-Judge variants vs Alakazam 1074 (200 BO3)

All runs: standard PPO, 20 processes × 5 BO3 × 2 orders = 200 BO3.

| Variant | Changes vs `hydrapple_dipam` | Hydra W-L | Hydra BO3 WR |
|---|---|---:|---:|
| baseline | — | 33-167 | **16.5%** |
| `hydra_nojudge` | −1 Judge, +1 Dawn | 10-190 | **5.0%** |
| `hydra_speed_search` | +2 Dawn, +2 Poké Pad; −1 Judge, −1 Lana's Aid, −1 Night Stretcher, −1 Tapu Bulu | 12-188 | **6.0%** |
| `hydra_speed_draw` | +1 Dawn, +1 Poké Pad, +1 Meowth ex; −1 Judge, −1 Lana's Aid, −1 Tapu Bulu | 5-195 | **2.5%** |
| `hydra_speed_feza` | +2 Dawn, +1 Poké Pad, +1 Fezandipiti ex; −1 Judge, −1 Lana's Aid, −1 Night Stretcher, −1 Tapu Bulu | 8-192 | **4.0%** |

All speed/no-Judge variants are far worse than the 16.5% baseline.

Why:

- The single Judge is real disruption; removing it lets Alakazam reach 17-19
  cards more often.  The 3-Judge version was also worse, so the issue is not
  "more Judge"; it is that the matchup is structural.
- The cuts used to make room (Lana's Aid, Night Stretcher, Tapu Bulu) removed
  recovery and the single-prize attacker; Hydrapple cannot sustain the long
  game without them.
- The base is already fast: 4 Forest of Vitality, 4 Bug Catching Set, 4 Ultra
  Ball, 4 Lillie's, 2 Dawn, 2 Poké Pad.  Adding more search does not fix the
  prize trade; Alakazam's single-prize engine still wins.
- The best Hydrapple list remains the original `hydrapple_dipam` at 16.5% BO3.

## Unfair legality note

Unfair Stamp can only be used if one of your Pokémon was Knocked Out during
the opponent's last turn.  In the sampled Alakazam games the Unfair-legal
windows were mostly at Alakazam hand 9-14, with rare 15-16 and one 19.  Holding
it for a 17+ burst therefore usually means the card is dead; the +17 rule was
broken by the legality condition, exactly as suspected.

## Bottom line

Hydrapple cannot be fixed against Alakazam 1074 by item swaps, Unfair timing,
Judge count, or speed cards.  The matchup is structural.  Alakazam 973.9 is
the clean ladder answer; Hydrapple should be treated as a good generalist that
has a hard counter, not as a deck to solve Alakazam with.
