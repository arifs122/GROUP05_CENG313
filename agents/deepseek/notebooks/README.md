# Grass/Ogerpon training package

## Dataset

`../data/dataset_grass_ogerpon.npz`

- Deck: `../decks/grass_ogerpon_kangaskhan.csv` — the exact 60-card Ogerpon /
  Mega Kangaskhan / Latias shell used by the #2 ladder team.
- Source replays: `replays/top2_policy`, `replays/top2_more`, `replays/top5_nov1`
  (deduped by EpisodeId).
- 332 matched episodes, 822 games, **68.671 karar**.
- Split: 55.199 train / 13.472 val (episode-based).
- Features: `state=78`, `option=68`, `max_options=40`, `card_ids=[40,2]`.
- Labels: game/round win (0/1, draw=0.5), single-select action index,
  selected type/context/maxCount.

Only public/own observation fields are used for features. `visualize` is used
only to identify which player is piloting the selected deck; it is not a model
feature. Opponent hand/prize identities and `logs` are not used.

## Notebook

`train_grass_bc.ipynb`

- Adds the dataset as a private Kaggle input and runs all.
- Trains:
  - value net: 78 -> 64 -> 32 -> 1, BCE;
  - policy prior: state-conditioned one-block attention pointer scorer.
- Policy loss excludes ATTACH/target decisions via zero sample weight.
- Exports `value_model_grass.npz` and `policy_model_grass.npz` to
  `/kaggle/working`.
- Runs PyTorch -> NumPy parity checks.
- Does not create or submit a Kaggle submission.

Reference run on the full dataset (CPU, 2026-11-05):

- value: val BCE ~0.618, AUC ~0.740, acc ~0.647
- policy: val top-1 ~0.526, top-3 ~0.828 (winner, single-select rows)
- parity: value ~5e-7, policy ~2e-6 max abs diff

Model files from that run are also under `../models/*_grass.npz`.
