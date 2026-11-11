# GROUP05_CENG313 — TransforMew

**CENG313 — Introduction to Data Science · Course Project**

Pokémon TCG AI Battle Challenge agent, replay analysis and the Assignment 1
EDA/baseline notebook.

| | |
|---|---|
| **Group ID** | **05** |
| **Kaggle team** | Attention Is All You Mew |
| **Competition** | [The Pokémon Company — PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground) |
| **Repository** | `ceng313-2026/GROUP05_CENG313` (course organization) |
| **Assignment 1 tag** | `assignment1` |
| **Assignment 1 deadline** | 13 November 2026, 23:59 |
| **Assignment 2 deadline** | 30 December 2026, 23:59 |

## Team

| Member | GitHub | Student ID |
|---|---|---|
| Arif Dayı | [@arifs122](https://github.com/arifs122) | 24118080010 |
| Fariz Berke Tutanç | [@Farhandir](https://github.com/Farhandir) | 24118080063 |
| Yunus Erdem Ergül | [@qewer33](https://github.com/qewer33) | 24118080078 |

Every member has commits in this repository; see **Insights → Contributors**.

## Competition and task

The competition is a **simulation** competition, not a tabular Kaggle
competition. Each submission is an agent. At every decision the simulator sends
an observation with the board state and a list of legal options, and the agent
returns the indices of the options it selects. Matches are played in
**best-of-three (BO3)** format and Kaggle evaluates the agents with a
Bradley–Terry skill rating computed from many episodes.

There is no `train.csv`. The observed data are the episode replays: each replay
contains both 60-card decks, the result and every decision. We use those
replays for the EDA and the baselines.

### Current score (our own agents)

| Submission | Member | Date | Public score | Role |
|---|---|---:|---:|---|
| `sample_submission.tar.gz` | Yunus | 2026-10-21 | 89.5 | sample control |
| `dragapult.tar.gz` | Arif | 2026-10-21 | **521.1** | **Assignment 1 baseline** (our rule policy) |
| `zipzop_baseline.tar.gz` | Arif | 2026-11-05 | 556.6 | improved rule baseline (ours) |
| `dragapult.tar.gz` (Dragapult2) | Yunus | 2026-10-21 | 595.1 | early Dragapult rule bot (ours) |
| `tokatci_lean4_noai.tar.gz` | Arif | 2026-11-03 | 611.5 | Lean4 rule/search agent (ours) |
| `mazlum_v21_matchups.tar.gz` | Fariz | 2026-11-06 | 622.6 | matchup-targeted Mazlum BC (ours) |
| `mazlum.tar.gz` | Fariz | 2026-11-03 | **643.1** | **best own learning agent** (Mazlum BC) |

Our own best submission is **643.1**; the top public score is **1573.9**, so
our best reaches about **41%** of the top score and the Assignment 1 baseline
reaches **33%**. The score proof only includes our own agents:
[`submissions/kaggle_submissions.png`](submissions/kaggle_submissions.png)
and [`submissions/kaggle_submissions.csv`](submissions/kaggle_submissions.csv).

> **Provenance.** The rule/Lean4/search bots and the Mazlum
> behavioural-cloning models are **our own code**. Assignment 2 will document
> our own improved implementation and cite any adapted idea.

### Team contributions

| Member | Main contributions |
|---|---|
| **Arif Dayı** | Rule-based and Lean4/search agents, submission pipeline, replay analysis, Assignment 1 notebook and `src/` modules |
| **Fariz Berke Tutanç** | Mazlum behavioural-cloning / value models, Banette and Slowking specialists, replay collection and matchup analysis |
| **Yunus Erdem Ergül** | Early Dragapult bot and agent-last validation fix, arena `reset()` hook, `tools/analyze.py`, bot contract and legality tests |

The detailed commit history is visible in **Insights → Contributors**. Every
member has commits in the repository.

## Required repository structure

```
GROUP05_CENG313/
├── README.md                          ← this file
├── requirements.txt                   ← Python packages used
├── .gitignore                         ← excludes data/ and replays/
├── assignment1_eda_baseline.ipynb     ← Assignment 1: EDA & baseline
├── assignment2_solution.ipynb         ← Assignment 2: improved solution (WIP)
├── src/
│   ├── __init__.py
│   ├── data.py                        ← loading & preprocessing functions
│   ├── models.py                      ← baseline training & evaluation
│   └── utils.py                       ← plots, confidence intervals, proof
├── data/                              ← NOT committed (competition data)
├── replays/                           ← NOT committed (competition data)
└── submissions/                       ← score proof + submission history
```

The repository also keeps the original **TransforMew** agent development
directories (`bots/`, `benchmarks/`, `decks/`, `tools/`, `src/agent/`,
`src/mazlum/`). They are part of the project and are used by Assignment 2; the
required course files above are the entry points for grading.

## Assignment 1: EDA & baseline

**Notebook:** [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb)
(saved with outputs, so it can be read on GitHub).

### What it contains

1. **Problem definition** — agent task, BO3 match target, Bradley–Terry metric.
2. **Data overview** — 5,879 replay matches, 11,758 deck appearances, 1,431
   cards, missing values, duplicates and deck-length checks.
3. **EDA with interpretations** — target balance, replay-source bias,
   archetype popularity and win rates with Wilson intervals, matchup heatmap,
   card-kind composition, staple cards, first-player choice, exact-list
   concentration, outliers.
4. **Preprocessing** — player-perspective rows, group split by `episode_id`,
   one-hot matchup/archetype features, card-composition features, scaling and
   leakage checks.
5. **Baselines** — logistic regression match-outcome model and the rule-based
   agent; local arena evaluation and Kaggle scores.
6. **Kaggle submission** — baseline score 521.1, own best 643.1, proof
   image in `submissions/`.
7. **Conclusion** — findings, limitations and the Assignment 2 plan.

### Headline results

| Baseline | Result |
|---|---|
| Dummy (training prior) | 50.0% match-winner accuracy |
| Logistic regression (deck/matchup features) | **62.0%** match-winner accuracy, **ROC-AUC 0.669**, log loss 0.643 |
| Matchup-only ablation | **61.4%** match-winner accuracy, **ROC-AUC 0.663** |
| Rule agent vs random legal actions | **80.5%** win rate (3 × 500 games) |
| Rule agent vs first-legal-action agent | **50.9%** win rate (3 × 500 games) |
| Kaggle rule baseline | **521.1** public score |
| Kaggle own best (Mazlum BC) | **643.1** public score |

The match-outcome model is **not** the submitted agent. It shows that deck and
matchup features carry real signal and gives Assignment 2 a validated feature
set; the agent baseline is evaluated separately in the arena and on Kaggle.

### Key EDA findings

* The meta is concentrated: Dragapult is 40.6% of deck slots; the top five
  exact lists are 43.9% of all decks and the top twenty are 80.2% (348 distinct
  lists in 11,758 deck slots).
* Matchups dominate archetype averages: Mega Excadrill beats Mega Kangaskhan
  79.9% but loses to Dragapult 65.5%; Dragapult beats Alakazam 66.3% but loses
  to Mega Kangaskhan 40.4%.
* A typical deck has ~18.8 Pokémon, 15.5 items, 11.6 supporters and 11.5
  energy; Lillie's Determination, Buddy-Buddy Poffin and Ultra Ball are the
  most played cards.
* The first player wins 51.2% of decisive matches (95% CI 49.9–52.5%), so going
  first is a strategic choice in this sample, not a measurable advantage.

## Data and setup

### 1. Competition data (not committed)

The engine and card data are competition-use-only and must stay out of git.
Download the competition data after accepting the rules:

```bash
# Kaggle CLI: https://github.com/Kaggle/kaggle-cli
kaggle competitions download \
  -c the-pokemon-company-ptcg-ai-battle-challenge-playground \
  -p data/raw
unzip data/raw/the-pokemon-company-ptcg-ai-battle-challenge-playground.zip -d data/raw
```

The archive contains the sample submission (`cg/`) and the card CSVs. Copy the
engine and card data into the git-ignored folders:

```bash
bash scripts/setup_assets.sh /path/to/sample_submission/cg /path/to/card/csvs
python scripts/check_env.py
```

`scripts/setup_assets.sh` copies the engine to `vendor/cg/` and the card CSVs
to `data/`. The local arena needs `vendor/cg/`; the notebooks need
`data/EN_Card_Data_R2_full.csv`.

### 2. Replay data (not committed)

Place the episode replay JSON files under `replays/`. Replays can be downloaded
from the Kaggle episode/submission pages and from the agent logs. The folder is
git-ignored because it is competition data and can be tens of GB.

Build the compact analysis table used by the notebook:

```bash
python tools/build_master_replays.py --out-dir data
python tools/build_analysis_dataset.py
```

This writes `data/master_replays.csv` and `data/analysis_matches.csv` (both
ignored). `build_analysis_dataset.py` also extracts the first player from the
beginning of each replay and adds deck-composition features.

### 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` contains the analysis stack (`numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `jupyter`, `nbconvert`). The Kaggle submission
bundle uses only NumPy inference; training dependencies are listed separately in
`requirements-train.txt`.

### 4. Run the notebooks

The committed notebook already contains its outputs. To reproduce them:

```bash
jupyter nbconvert --to notebook --execute assignment1_eda_baseline.ipynb --inplace
```

or open it with JupyterLab and **Restart & Run All**.

## Kaggle submission

The submission bundle is a `.tar.gz` with `main.py`, `deck.csv` and `cg/` at the
top level. To build and submit a bot:

```bash
python tools/build_submission.py --bot bots/<bot_name>
kaggle competitions submit \
  -c the-pokemon-company-ptcg-ai-battle-challenge-playground \
  -f submissions/<bot_name>.tar.gz \
  -m "description"
```

The competition allows 5 submissions per day and keeps the latest 2 active.
The Assignment 1 baseline is the rule policy in `src/agent/`; the best own
learning agent is the Mazlum behavioural-cloning line (643.1).

## Original TransforMew development

The repository started as **TransforMew**, a full agent-development project. It
contains:

| Path | Purpose |
|---|---|
| `src/agent/` | baseline observation → action policy |
| `src/mazlum/` | behavioural-cloning / value-model training code |
| `bots/` | team bots, each with `main.py` + `deck.csv` |
| `decks/` | shared deck lists and meta-deck analyses |
| `tools/` | arena, tournaments, replay parsing, deck checks, submission builder |
| `benchmarks/` | reference agents and meta decks |
| `scripts/` | environment and asset setup |
| `tests/` | smoke tests |

Useful commands:

```bash
python tools/arena.py --games 100
python tools/arena.py --bot0 bots/<a> --bot1 bots/<b> --games 50 --bo3
python tools/analyze.py --bot bots/dragapult --all --games 100
python tools/tournament.py --games 50
python tools/check_deck.py bots/<bot>/deck.csv --engine
```

## Assignment 2 plan

The placeholder notebook is
[`assignment2_solution.ipynb`](assignment2_solution.ipynb). Planned work:

1. Board-state feature extraction from replay decisions.
2. Compare the calibrated matchup model with a board-state value model.
3. Behaviour cloning, then PPO fine-tuning against a league.
4. Matchup-aware deck choice instead of one fixed list.
5. BO3 arena evaluation with bootstrap intervals and a final Kaggle submission.

## Academic integrity

* All notebook cells and `src/` code in this repository were written by the
  group. We did not copy Kaggle notebooks or cells.
* Public notebooks/discussions were read for ideas only; any adapted idea will
  be cited in a markdown cell with a link.
* Competition data (`data/`, `replays/`, engine) is not committed.
* Every member can explain the cells they contributed and has meaningful
  commits in the repository.

## License

The original code in this repository is released under the MIT License (see
`LICENSE`). Pokémon cards, the `cabt` engine and the competition data are not
covered by that license; they remain subject to the Kaggle competition rules
and are not redistributed here.
