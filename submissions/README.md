# submissions/

This folder contains the score proof required by Assignment 1 and the
submission history captured from Kaggle.

| File | What it is |
|---|---|
| `kaggle_submissions.png` | Rendered capture of the team's Kaggle **My Submissions** page (via the Kaggle CLI), annotated with the member and the provenance of each run. |
| `kaggle_submissions.csv` | The same history in machine-readable form: ref, member, file, date, description, status, public score. |
| `*.tar.gz` | Submission bundles. They are git-ignored because they contain the competition SDK; only the small score proof is committed. |

## Provenance

- **Our own work:** the rule/Lean4/search bots, the Mazlum behavioural-cloning
  / value models, the EDA notebook and the `src/` modules.
- **Assignment 1 baseline:** `dragapult.tar.gz` (Kaggle description
  "Dragapult sample policy v1"), public score **521.1**.
- The reported table contains only our own agents. Exploratory PPO runs are
  not included in the reported results and are not claimed as our solution.

## How the proof was captured

```bash
kaggle competitions submissions \
  -c the-pokemon-company-ptcg-ai-battle-challenge-playground \
  -v > submissions/kaggle_submissions_raw.csv
python -c "from src.utils import load_submission_history, render_submission_proof; render_submission_proof(load_submission_history())"
```

The committed CSV is a curated excerpt: sample control, the Assignment 1 rule
baseline, and our own rule/BC agents.

## Note on re-submitting

The competition keeps only the **latest two** active submissions. A new weak
baseline submission would displace one of the stronger agents, so the team did
not re-submit the Assignment 1 baseline. The proof above is the original
baseline submission from 2026-10-21.
