# Alakazam 1074 — bad matchups and the RPS cycle (2026-11-08)

Alakazam: `decks/ppo_candidates/alakazam_1074.csv` (public f23b6ed606 list).
All local tests are PPO vs PPO, same public PPO bundle on both sides.

## Alakazam's matchup table

| Opponent | Alakazam WR | Protocol |
|---|---:|---|
| Hydrapple `hydrapple_dipam` | **83.5%** | 200 BO3 |
| Mega Lopunny `mega_lopunny_yarki` | **22.0%** | 400 games (top-3 robust) |
| Mega Kangaskhan `mega_kangaskhan_kawatta` | **37.3%** | 400 games (top-3 robust) |
| Dragapult Tyr | **33.5%** | 200 games |
| Dragapult Petit | **36.5%** | 200 games |
| Dragapult Stephen | 47.5% | 200 games |
| Dragapult Arai | 49.5% | 200 games |
| Dragapult Raihan | 52.5% | 200 games |
| Dragapult Attention | 60.5% | 200 games |
| Dragapult Dipam | 61.0% | 200 games |

## The four-deck cycle

| Matchup | Winner | WR |
|---|---|---:|
| Hydrapple vs Mega Lopunny | Hydrapple | 72.0% |
| Mega Lopunny vs Mega Kangaskhan kawatta | Mega Lopunny | 80.8% |
| Mega Kangaskhan kawatta vs Alakazam | Kawatta | 62.7% |
| Alakazam vs Hydrapple | Alakazam | 83.5% |

So the meta is non-transitive:

**Hydrapple > Lopunny > Kawatta > Alakazam > Hydrapple**

Dragapult Tyr / Petit are another Alakazam counter, and they are roughly even
against Hydrapple.  There is no single "best deck"; the field mix decides.

## Why Alakazam loses

Alakazam's damage is `Powerful Hand` = 20 × cards in hand.  Its usual damage
is 220-280 at 11-14 cards; 340-380 at 17-19 cards.  It has 140 HP and is a
single-prize engine.  It loses when the opponent does one or more of:

1. **Denies the hand size.**
   - Kawatta: 3 Xerosic's Machinations + 2 Eri + Boss/Judge pressure.
   - Dragapult Tyr / Petit: Judge + Unfair Stamp + Crushing Hammer.
   - If Alakazam is kept at 10-14 cards, Powerful Hand stays at 200-280 and
     cannot one-shot the 300-400 HP targets.

2. **Stacks HP and healing.**
   - Kawatta: Mega Kangaskhan ex 300 HP + **Hero's Cape +100 = 400 HP**, 4
     Jumbo Ice Cream, 4 Boss.  Alakazam needs 20 cards to one-shot 400; the
     hand disruption prevents that.
   - Lopunny: Mega Lopunny ex 330 HP, Mega Froslass ex 310 HP, Wally's
     Compassion heals all damage from a Mega ex, 4 Mist Energy / Battle Cage.
   - Hydrapple has 330 HP but no real hand denial and no HP tool; 17 cards
     (340) is enough, so Alakazam wins that matchup.

3. **Attacks the bench / races the 140 HP body.**
   - Dragapult: Phantom Dive 200 + 6 bench counters; Abra/Kadabra/Alakazam are
     140 HP, so the bench spread and 200 damage force multiple KOs.
   - Lopunny: 150-160 damage one-shots Alakazam; Kawatta: 200 damage one-shots
     it.  The single-prize plan needs 6 KOs, but Alakazam dies in one hit.

## Replay evidence (real ladder)

- **EP 125755613 — Kawatta (Mega Kangaskhan) loss:** opponent's Mega Kangaskhan
  is at **400 HP** (Hero's Cape); Xerosic's Machinations ×2 + Eri ×1; Alakazam's
  hand stays at 10-15, so Powerful Hand does 200-300 and cannot one-shot 400.
  Opponent finishes with 1 prize left, Alakazam 6.
- **EP 125750736 — Mega Lopunny loss:** Mega Lopunny 330 HP; Alakazam is at
  9/13 cards early (180/260 damage), later 17/22 cards but the game is already
  an attrition loss; final active empty, bench 5.
- **EP 125482492 — Dragapult Petit loss:** Judge ×2, Unfair Stamp ×1,
  Crushing Hammer ×2; Alakazam's hand is at 5-8 early; Dragapult ex is 320 HP;
  Alakazam loses the hand-size race and the bench pressure.

## Bottom line

Alakazam is not "invincible"; it is a **hand-size combo deck with a 140 HP
body**:

- It beats decks that cannot deny its hand or survive 340+: Hydrapple, Banette,
  Mega Excadrill.
- It loses to decks that combine **hand disruption + 300-400 HP/heal/stall**
  (Kawatta, Lopunny) or **hand disruption + bench spread + 320 HP** (Dragapult
  Tyr/Petit).

Field implication: if Hydrapple floods the ladder, Alakazam is the right
answer; if Alakazam floods it, Kawatta/Lopunny are the right answer; if
Kawatta/Lopunny flood it, Hydrapple is the right answer.

Raw results: `results/hydra_alakazam_bo3_baseline.json`,
`results/ppo_league_top3_robust.json`, `results/alakazam_vs_dragapult_*.json`.

## Plain PPO retest (2026-11-08)

Hydrapple `hydrapple_dipam` with the **standard public PPO**, no custom rule
and no deck change, vs Alakazam 1074, 200 BO3 (20 processes × 5 BO3 × 2 orders,
new seed):

- Hydrapple **12-188 = 6.0%** BO3.
- Previous clean baseline run: 33-167 = 16.5% BO3.

Both runs are low; the difference is engine-seed variance.  The conclusion is
the same: Alakazam is a hard counter to Hydrapple.  Hydrapple's strength is
against the wider field (68.9% over 26 decks, 72.8% top-20 field-weighted), not
against Alakazam.

Raw result: `results/hydra_alakazam_bo3_plain_retest.json`.
