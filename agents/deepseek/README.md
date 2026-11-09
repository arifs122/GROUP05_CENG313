# DeepSeek workspace

Bu klasör DeepSeek ajanının izole çalışma alanıdır. Diğer ajanların ve mevcut
`bots/` botlarının dosyalarına dokunulmaz.

## Bot

`top2_core/` — sıfırdan Top2 Dragapult ajanı:

- `main.py` — Kaggle/arena girişi
- `top2_core_policy.py` — kural + arama + model orkestrasyonu
- `top2_core_rules.py` — Katman 0/1: legal kural çekirdeği ve hasar sayacı
- `top2_core_search.py` — Katman 2: determinize 1-ply arama
- `top2_core_features.py` — eğitim/çıkarım ortak özellik çıkarımı
- `top2_core_value.py` — NumPy value net çıkarımı
- `top2_core_policy_net.py` — NumPy policy prior çıkarımı
- `deck.csv` — Top2 listesi (60 kart)

Modeller `models/` altında; bot `top2_core/` içinde de aranır.

## Veri

`training/build_dataset.py` yalnızca Petit Canard gözlemlerini kullanır.
`visualize`, `search_begin_input`, rakip eli/prize kimlikleri ve `logs`
feature olarak kullanılmaz. Split episode bazlıdır.

```bash
.venv/bin/python agents/deepseek/training/build_dataset.py \
  --out agents/deepseek/data/dataset_top2_full.npz
```

## Eğitim

```bash
.venv/bin/python agents/deepseek/training/train_value.py \
  --dataset agents/deepseek/data/dataset_top2_full.npz \
  --hidden 64 --dropout 0.2 --lr 3e-4 --batch 2048

.venv/bin/python agents/deepseek/training/train_policy.py \
  --dataset agents/deepseek/data/dataset_top2_full.npz
```

Her iki script de PyTorch -> NumPy parite testini çalıştırır.

## Test

```bash
.venv/bin/python tools/arena.py \
  --bot0 agents/deepseek/top2_core \
  --bot1 bots/_template \
  --games 50 --seed 31
```

Submission üretimi/atımı yalnızca kullanıcı isterse yapılır.
