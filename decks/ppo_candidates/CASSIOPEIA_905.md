# Cassiopeia (Hydrapple PPO) — 905.3 ladder review

Submission `58124604`, `cassiopeia.tar.gz`, description `cassiopeia ppo`.
PPO bundle: standard public PPO, deck `hydrapple_dipam`, no custom rules.

Sample: **153 public episodes** downloaded to `replays/cassiopeia_58124604`.

## Overall

- **81-72 = 52.9%** across the sample.
- First: 48-39 = 55.2% (87 games).
- Second: 33-33 = 50.0% (66 games).
- Average 367 steps, 2.45 rounds.
- All 153 episodes `DONE`; no errors/timeouts.
- Public score: **905.3**.

## Field at ~905

| Opponent archetype | Games | Our W-L | Our WR |
|---|---:|---:|---:|
| **Hydrapple (mirror)** | **87** | 43-44 | **49.4%** |
| Alakazam | 23 | 8-15 | **34.8%** |
| Dragapult | 16 | 12-4 | **75.0%** |
| Festival Lead | 6 | 3-3 | 50.0% |
| Mega Kangaskhan | 5 | 3-2 | 60.0% |
| Mega Lopunny | 5 | 4-1 | **80.0%** |
| Mega Lucario | 4 | 1-3 | **25.0%** |
| Slowking-Kangaskhan | 2 | 2-0 | 100% |
| Mega Abomasnow | 2 | 2-0 | 100% |
| Other (Dwebble / Archaludon / Garchomp) | 3 | 3-0 | 100% |

## Mirror detail

Of the 87 Hydrapple mirror games, **77 were the exact same 60-card list as
ours**; we went **40-37 = 51.9%** in those. The other 10 mirror games were
variant lists; we went 3-7.  So the mirror is essentially a coin flip, and it
is **57% of the field**.

## Reading

- The 905 score is mostly a **mirror meta** result: 57% of games are the same
  Hydrapple list at ~50%, plus a strong edge against Dragapult (75%) and Mega
  Lopunny (80%).  Alakazam (34.8%) is the main drag.
- The local `HYDRAPPLE_VS_FIELD` number (68.9% over 26 decks) does **not**
  transfer to the 905 ladder: the actual field is Hydrapple mirror + Alakazam,
  not the top-20 list distribution (Mega Excadrill/Banette-heavy).  The real
  field is much more hostile.
- The local Alakazam hard-counter result is confirmed: 8-15 = 34.8%, and the
  most common Alakazam list held us to 3-11 / 27.3%.
- Mega Lucario is a small-sample warning: 1-3, with a 2-game list at 0-2.  It
  is not a common field deck yet, but it is another bad matchup.
- First/second is not the issue: 55.2% first / 50.0% second.  The deck is
  stable; no crash/fallback.
- Trend by episode id: 55.3% → 42.1% → 60.5% → 55.3%.  The score is a plateau,
  not a climb; the mirror keeps it near 900.

## What this means

- **Cassiopeia 905.3 is a solid generalist**, the best of the new PPO
  submissions (shaco 883.5, gragas 862.1), but still below the public
  **Alakazam 973.9**.
- The field is Hydrapple-heavy, so **Alakazam is the clean counter** (local
  Alakazam vs Hydrapple 83.5% BO3, ladder Alakazam 973.9).  If the goal is to
  climb above 905, Alakazam is the better main; Hydrapple is capped by the
  mirror and Alakazam.
- If Hydrapple is kept, the only meaningful lever is a **mirror edge**: 57% of
  games are the exact same list at ~52%.  A mirror-specific tech could move the
  score, but the item/Judge variants tested locally were worse against
  Alakazam, so any change must be A/B tested against both the mirror and
  Alakazam.

Raw replays: `replays/cassiopeia_58124604` (153 episodes).

## Non-Hydrapple field at ~905

Total 153 games: Hydrapple 87 (56.9%), non-Hydrapple 66 (43.1%).

| Non-Hydrapple deck | Games | Share of non-Hydra | Our W-L | Our WR |
|---|---:|---:|---:|---:|
| **Alakazam** | 23 | **34.8%** | 8-15 | 34.8% |
| **Dragapult** | 16 | **24.2%** | 12-4 | 75.0% |
| Festival Lead | 6 | 9.1% | 3-3 | 50.0% |
| Mega Lopunny | 5 | 7.6% | 4-1 | 80.0% |
| Mega Kangaskhan | 5 | 7.6% | 3-2 | 60.0% |
| Mega Lucario | 4 | 6.1% | 1-3 | 25.0% |
| Slowking-Kangaskhan | 2 | 3.0% | 2-0 | 100% |
| Mega Abomasnow | 2 | 3.0% | 2-0 | 100% |
| Other (Dwebble/Garchomp/Archaludon) | 3 | 4.5% | 3-0 | 100% |

So after Hydrapple, the ladder is mostly **Alakazam (35% of non-Hydra) +
Dragapult (24%)**.  Everything else is scattered.

### Exact opponent lists

- Hydrapple: **77/87 exact `hydrapple_dipam`**, 10 variants; mirror W-L 40-37 = 51.9%.
- Alakazam: **11/23 exact `alakazam_1074`** (the f23b6ed606 list), 10 variants, 2 generic; vs the exact list we were 3-8 = 27.3% (Alakazam 72.7%).
- Dragapult: 7 `dragapult_dipam`, 4 `dragapult_petit`, 3 variants, 1 Arai, 1 Raihan; our 12-4 = 75%.
- Festival: 6/6 exact `festival_lead`.
- Mega Kangaskhan: 2 yarki + 2 kawatta + 1 variant.
- Mega Lopunny: 3 yarki + 2 variants.
- Mega Lucario: 4 variants.

## Why is Hydrapple everywhere?

- It is the **public 2nd-place final list**, easy to copy, and it is strong with
  the public PPO: it beats Dragapult/Lopunny/Kangaskhan and goes ~50% in the
  mirror.
- The mirror is a low-variance safe choice; at 905, 57% of games are Hydrapple,
  and 77/87 mirror games are the exact same list.  The ladder self-reinforces:
  players copy the top public list, it does not collapse, so it stays.
- The counter is **Alakazam** (973.9 public score): it farms Hydrapple (local
  83.5% BO3; ladder exact-list sample 72.7%), but it is less common and harder
  to pilot than the Hydrapple netdeck.  The 905 field is essentially a
  two-deck meta: Hydrapple + Alakazam.

## Kassadin (Alakazam) submission

- Ref `58161343`, `kassadin.tar.gz`, description `kassadin ppo`.
- Standard public PPO + `alakazam_1074` (same f23b6ed606 list as the existing
  973.9 submission).
- Initial public score: 600.0 (provisional; will climb as matches accumulate).
- Expected field: Hydrapple ~57% (good for Alakazam), Alakazam mirror ~15%
  (coin flip), Dragapult ~10% (mixed: `dragapult_dipam` good, `dragapult_petit`
  bad), the rest scattered.  The existing Alakazam at 973.9 is the reference;
  Kassadin is a fresh duplicate run.
