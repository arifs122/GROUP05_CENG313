# submissions/

This folder contains the score proof required by Assignment 1 and the
submission history captured from Kaggle.

| File | What it is |
|---|---|
| `kaggle_submissions.png` | Rendered capture of the team's Kaggle **My Submissions** page (via the Kaggle CLI). The full-resolution image is the score proof. |
| `kaggle_submissions.csv` | The same history in machine-readable form: ref, file, date, description, status, public score. |
| `*.tar.gz` | Submission bundles. They are git-ignored because they contain the competition SDK; only the small score proof is committed. |

## How the proof was captured

```bash
kaggle competitions submissions \
  -c the-pokemon-company-ptcg-ai-battle-challenge-playground \
  -v > submissions/kaggle_submissions.csv
python -c "from src.utils import load_submission_history, render_submission_proof; render_submission_proof(load_submission_history())"
```

The CSV above is a curated excerpt containing the sample control, the
Assignment 1 rule baseline (`dragapult.tar.gz`, 521.1), the improved rule
baseline (`zipzop_baseline.tar.gz`, 556.6) and the current best PPO agent
(`kassadin.tar.gz`, 1035.4).

## Note on re-submitting

The competition keeps only the **latest two** active submissions. A new weak
baseline submission would displace one of the strong agents, so the team did
not re-submit the Assignment 1 baseline. The proof above is the original
baseline submission from 2026-10-21.
