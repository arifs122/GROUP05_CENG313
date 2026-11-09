# PPO candidate decks

This folder contains the deck lists used by `tools/ppo_league.py`.  The point is
to compare **the same PPO policy** on different 60-card lists; no rule-based or
random local bot is involved.

Lists fall into two groups:

- **Top-20 replay lists**: exact 60-card lists extracted from
  `results/master_replays.csv`.  The source team and sample size are listed
  below.  Some teams share the same list; identical lists are stored once.
- **Existing meta lists**: consensus/testing lists already in `decks/meta/`
  (festival, archaludon, teal mask, crustle variants, mega starmie, N's Zorua).

## Top-20 extracts

| File | Source team | Games in corpus | Win rate |
|---|---|---:|---:|
| `dragapult_attention.csv` | Attention Is All You Mew | 792 | 47.9% |
| `banette_attention.csv` | Attention Is All You Mew | 309 | 48.9% |
| `mega_kangaskhan_yarki.csv` | やる気元気ミワハルキ 2nd ver. | 754 | 67.8% |
| `mega_lopunny_yarki.csv` | やる気元気ミワハルキ 2nd ver. | 462 | 56.1% |
| `slowking_kangaskhan_yarki.csv` | やる気元気ミワハルキ 2nd ver. | 35 | 31.4% |
| `dragapult_dipam.csv` | Dipam Chakraborty | 1131 | 55.2% |
| `hydrapple_dipam.csv` | Dipam Chakraborty | 12 | 50.0% |
| `dragapult_petit.csv` | Petit Canard | 916 | 59.6% |
| `mega_excadrill_knshnb.csv` | knshnb | 629 | 57.6% |
| `dragapult_raihan.csv` | Raihan Ramadistra | 514 | 55.4% |
| `mega_excadrill_verbena.csv` | バーベナヘレナでコンボ決めたい連合 | 306 | 64.7% |
| `mega_kangaskhan_kawatta.csv` | KawattaTaido | 380 | 54.7% |
| `dragapult_rtoabc2.csv` | RtoABC2 | 52 | 36.5% |
| `dragapult_arai.csv` | アライ塾Kaggle支部 | 141 | 58.2% |
| `mega_excadrill_seagoat.csv` | SeaGoat | 171 | 46.2% |
| `dragapult_tyr.csv` | tyr1218 | 135 | 42.2% |
| `dragapult_stephen.csv` | Stephen Schott | 37 | 40.5% |
| `mega_excadrill_ondrej.csv` | Ondrej Kubicek (variant) | 39 | 79.5% |

`mega_excadrill_ondrej.csv` is the 39-game variant, not Ondrej's most common
list (that one is identical to `mega_excadrill_knshnb.csv`).

## Existing meta lists

`alakazam_1074.csv` is the public f23b6ed606 Alakazam list (the 1074.1 public
agent).  `alakazam_generic.csv` is the older generic Alakazam list kept as a
control.  The rest are the usual `decks/meta/` lists.

## Run the league

```bash
# all candidate decks, both deck orders, 15 games per cell
.venv/bin/python tools/ppo_league.py \
  --decks decks/ppo_candidates --games 15 --workers 6 \
  --out results/ppo_league.json
```

For a quick sanity check:

```bash
.venv/bin/python tools/ppo_league.py \
  --decks decks/ppo_candidates/alakazam_1074.csv \
          decks/ppo_candidates/mega_excadrill_knshnb.csv \
          decks/ppo_candidates/dragapult_petit.csv \
  --games 5 --workers 2 --out results/ppo_league_smoke.json
```

The output JSON has the full pairwise matrix, game counts, overall win rates,
and a Bradley-Terry strength estimate.  The console prints the ranking and a
compact matrix.
