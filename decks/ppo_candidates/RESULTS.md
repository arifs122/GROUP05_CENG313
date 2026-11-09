# PPO deck league — 2026-11-07

All matches use the **same public PPO bundle** (`bots/ppo_public`) on both
sides.  No rule-based or random local bot is involved.  The only variable is
the 60-card list.

## Method

- **Screening:** 27 decks, 702 ordered pairs, 15 games per cell, both deck
  orders.  10,530 games, 26.5 min.  Useful for eliminating weak decks, but
  individual cells are noisy.
- **Top-3 validation:** 20 independent arena processes per ordered pair, 10
  games each (200 games/order, 400 games/matchup).  This is the reliable
  head-to-head table.
- **Focus-8:** 8 decks, 50 games/ordered pair.  Screen for the off-meta
  counters.

The local engine seeds its RNG per process.  One process with 100 games can
give a very different result from 20 processes with 10 games each.  Use
`tools/ppo_league.py --repeats N` for final decisions.

## Screening ranking (27 decks)

Field-weighted = expected win rate against the top-20 replay list
distribution, not a uniform field.  Values are approximate (15-game cells).

| Deck | Overall WR | Field-weighted top20 |
|---|---:|---:|
| `mega_kangaskhan_kawatta` | **78.5%** | **~81%** |
| `alakazam_1074` | 70.6% | ~71% |
| `mega_lopunny_yarki` | 68.5% | ~77% |
| `dragapult_tyr` | 65.1% | ~67% |
| `dragapult_arai` | 65.0% | ~64% |
| `dragapult_raihan` | 63.6% | ~62% |
| `archaludon` | 63.2% | ~68% |
| `dragapult_stephen` | 62.9% | ~64% |
| `teal_mask_ogerpon` | 62.8% | ~71% |
| `dragapult_petit` | 62.4% | ~63% |
| `hydrapple_dipam` | 62.2% | ~70% |
| `dragapult_rtoabc2` | 61.3% | ~61% |
| `dragapult_dipam` | 59.2% | ~66% |
| `alakazam_generic` | 59.0% | — |
| `festival_lead` | 55.9% | ~66% |
| `mega_starmie` | 54.1% | ~62% |
| `dragapult_attention` | 52.8% | — |
| `mega_kangaskhan_yarki` | 49.1% | — |
| `munkidori_crustle` | 41.0% | — |
| `crustle_fire` | 39.6% | — |
| `slowking_kangaskhan_yarki` | 34.9% | — |
| `ns_zorua` | 30.8% | — |
| `mega_excadrill_ondrej` | 21.5% | — |
| `banette_attention` | 19.5% | — |
| `mega_excadrill_knshnb` | 15.4% | — |
| `mega_excadrill_verbena` | 14.2% | — |
| `mega_excadrill_seagoat` | 13.5% | — |

**Mega Excadrill is not playable by this PPO policy.**  Even the best human
lists lose 79-86% of their games in the PPO mirror.  Fixing that requires
fine-tuning, not a deck swap.

## Robust top-3 (20 processes × 10 games per order)

| Matchup | Result |
|---|---:|
| Mega Lopunny vs Mega Kangaskhan | **80.8%** Lopunny |
| Mega Lopunny vs Alakazam 1074 | **78.0%** Lopunny |
| Mega Kangaskhan vs Alakazam 1074 | **62.7%** Kangaskhan |

In a pure three-deck meta: Lopunny 79.3%, Kangaskhan 41.1%, Alakazam 29.6%.
But the expected ladder field is not a three-deck meta; Lopunny has bad
matchups against several off-meta decks.

## Focus-8 (50 games per ordered pair)

| Deck | Overall WR |
|---|---:|
| `mega_kangaskhan_kawatta` | **72.4%** |
| `teal_mask_ogerpon` | 61.3% |
| `archaludon` | 58.3% |
| `hydrapple_dipam` | 53.7% |
| `mega_lopunny_yarki` | 47.0% |
| `dragapult_tyr` | 42.0% |
| `alakazam_1074` | 38.7% |
| `festival_lead` | 26.6% |

Key counter relationships in this sample:

- Kangaskhan beats Lopunny? No — the robust 400-game table says Lopunny
  wins ~81%.  The 50-game focus sample gave the opposite; treat the robust
  table as the truth.
- Lopunny loses to Archaludon (36.7%), Teal Mask (43.3%), Hydrapple (26.7%),
  Festival (26.7%).
- Kangaskhan beats Teal Mask, Archaludon, Hydrapple, Dragapult variants,
  Alakazam and Mega Excadrill; its only bad matchup is Lopunny.
- Teal Mask / Archaludon / Hydrapple are the off-meta counters to the
  Lopunny/Alakazam side of the meta, but they lose to Kangaskhan.

## Reading

1. **Kangaskhan `kawatta` is the most robust ladder generalist** for this
   PPO: highest field-weighted win rate against the top-20 distribution, and
   no bad matchup except Lopunny.
2. **Lopunny `yarki` is the best counter** to Kangaskhan and Alakazam, but it
   is a specialist: Archaludon / Teal Mask / Hydrapple / Festival punish it.
3. **Alakazam `1074` remains the proven ladder baseline** (1074.1 public
   reading) but is third in the PPO-only meta.
4. **Best Dragapult lists:** Tyr / Arai / Raihan / Petit.  The most common
   top-20 Dragapult list (Attention) is the worst of the PPO Dragapult
   variants.
5. **Mega Excadrill is off the table** unless the policy is fine-tuned.

## Suggested next step

- Main submission: `mega_kangaskhan_kawatta`.
- Hedge/second submission: `mega_lopunny_yarki` if the field is
  Kangaskhan/Alakazam-heavy; `teal_mask_ogerpon` or `archaludon` if the field
  is Dragapult/Lopunny-heavy.
- Keep `alakazam_1074` as the control until a ladder A/B beats 1074.1.
- Small deck changes should be A/B tested with
  `tools/ppo_league.py --repeats 20` against the top 4-6 field decks, not
  against random bots.

The raw results are in `results/ppo_league_big.json`,
`results/ppo_league_top3.json`, `results/ppo_league_top3_robust.json`,
`results/ppo_league_focus8.json`.

## Ladder replay check — 2026-11-07

Two PPO submissions were sent to the playground ladder and their recent public
episodes were downloaded (`replays/gragas_58097239`,
`replays/shaco_58097253`).  The public Alakazam submission
(`replays/alakazam1074_58030010`) is included as a baseline.

| Submission | Deck | Sample | Overall | vs Hydrapple | vs non-Hydrapple |
|---|---|---|---:|---:|---:|
| `gragas` | Mega Kangaskhan kawatta | 55 | 54.5% (30-25) | **25.9% (7-20)** | **82.1% (23-5)** |
| `shaco` | Mega Lopunny yarki | 55 | 58.2% (32-23) | **45.5% (10-12)** | 66.7% (22-11) |
| `alakazam1074` | Alakazam f23b6ed606 | 18 | 72.2% (13-5) | **71.4% (5-2)** | 72.7% (8-3) |

The decisive card is **Hydrapple**: 27/55 of gragas' sampled opponents and
22/55 of shaco's were Hydrapple, and 26/27 + 20/22 of those were the exact
`hydrapple_dipam` list.  That list uses non-ex Meganium / Tapu Bulu as well as
Hydrapple ex, so Kawatta's Crustle ex-immunity does not stop its main attacks.
Alakazam's scaling `Powerful Hand` does.

Other sampled matchups:

- **gragas:** Dragapult 5-2, Mega Kangaskhan 5-0, Alakazam 4-0, Mega Lopunny
  0-2, Hydrapple 7-20.  No errors/timeouts (all 55 `DONE`).
- **shaco:** Mega Kangaskhan 6-0, Mega Lucario 4-1, Mega Lopunny 2-0,
  Alakazam 3-4, Dragapult 3-4, Hydrapple 10-12.  No errors/timeouts.
- First/second: gragas 64.0% first vs 46.7% second; shaco vs Hydrapple 60.0%
  first vs 33.3% second.  The shipped agent already forces first; the rest is
  the coin flip.

Early public scores: gragas 688.9, shaco 600.0, Alakazam baseline 973.9.  The
scores are still settling; the sampled win rates are directional, not final.

