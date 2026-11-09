# Hydrapple vs the PPO field — 2026-11-08

Deck: `decks/ppo_candidates/hydrapple_dipam.csv` (the exact list seen in
20/22 of shaco's sampled Hydrapple ladder games).

Opponents: all 26 other decks in `decks/ppo_candidates/` (top-20 replay lists +
existing meta lists).  Protocol: **20 independent arena processes × 5 games ×
2 deck orders = 200 games per matchup**, both sides on the same public PPO
bundle.  Total: 5,198 games (2 draws).

## Overall

- Hydrapple: **3,584-1,614 = 68.9%** (equal-weight over the 26 opponents).
- Field-weighted against the top-20 replay list distribution: **72.8%**.
- Field-weighted against the latest-300 replay list distribution: **72.1%**.

## Matchups

Sorted by Hydrapple's win rate (higher = better for Hydrapple):

| Opponent | Hydra WR | Hydra W-L |
|---|---:|---:|
| mega_excadrill_ondrej | 100.0% | 200-0 |
| mega_excadrill_seagoat | 99.5% | 199-1 |
| mega_excadrill_knshnb | 99.0% | 198-2 |
| mega_excadrill_verbena | 98.5% | 197-3 |
| banette_attention | 97.0% | 194-6 |
| crustle_fire | 95.5% | 191-9 |
| ns_zorua | 95.0% | 190-10 |
| mega_starmie | 89.0% | 178-22 |
| slowking_kangaskhan_yarki | 80.0% | 160-38 |
| teal_mask_ogerpon | 79.5% | 159-41 |
| archaludon | 74.5% | 149-51 |
| mega_lopunny_yarki | 72.0% | 144-56 |
| munkidori_crustle | 71.5% | 143-57 |
| mega_kangaskhan_kawatta | 63.5% | 127-73 |
| dragapult_attention | 55.5% | 111-89 |
| dragapult_stephen | 55.5% | 111-89 |
| dragapult_raihan | 55.0% | 110-90 |
| mega_kangaskhan_yarki | 54.5% | 109-91 |
| dragapult_dipam | 53.0% | 106-94 |
| dragapult_petit | 52.5% | 105-95 |
| dragapult_tyr | 51.5% | 103-97 |
| dragapult_rtoabc2 | 48.5% | 97-103 |
| dragapult_arai | 45.5% | 91-109 |
| alakazam_generic | 45.0% | 90-110 |
| festival_lead | 36.0% | 72-128 |
| **alakazam_1074** | **25.0%** | **50-150** |

## Reading

- Hydrapple is a **dominant generalist** in this PPO field: it beats nearly
  everything except **Alakazam 1074** (Hydra 25% → Alakazam ~75%) and
  **Festival Lead** (Hydra 36% → Festival ~64%).  Dragapult is roughly even
  (45.5-55.5%).
- The ladder Hydrapple flood is rational: it destroys Mega Excadrill (~99%),
  Banette (97%), Crustle (95.5%), N's Zorua (95%), Mega Starmie (89%),
  Slowking (80%), Teal Mask (79.5%), Archaludon (74.5%), Mega Lopunny (72%),
  Munkidori-Crustle (71.5%) and Kawatta (63.5%).
- The field-weighted top-20 number (72.8%) is higher than the equal-weight
  number because the top-20 field is Mega Excadrill-heavy and Hydrapple farms
  that deck under PPO.
- **Alakazam 1074 remains the clean ladder answer**: single-prize, Powerful
  Hand one-shots 330 HP Hydrapple ex; the 200-game local test says Hydrapple
  wins only 25% against it, and the ladder sample had Alakazam 5-2 vs Hydrapple.
- Festival Lead is an interesting off-meta counter (Hydra 36%), but its
  overall PPO record is weak; it is a niche counter, not a main deck.
- Caveat: this is the local PPO field, not the ladder.  The candidate set has
  many weak decks and the local PPO Hydrapple pilot is strong.  Use the
  matchup table as a structural guide, not as expected ladder scores.
