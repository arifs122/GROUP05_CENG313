# Numpy ağları ve parite

## Policy prior

Eğitim: `python3 agents/mk_ogrepon/tools/train_prior.py`

- Veri: Top1 (`17014244`) 50 kazanılmış replay.
- `replay_meta.iter_decisions()`; sadece tek seçimli kararlar.
- 7.825 örnek (5.961 train / 1.864 val), replay bazlı split.
- Girdi: 112 boyutlu public state + 192 boyutlu option.
- Mimari: `(112+192) → 128 → 64 → 1`, ReLU, dropout 0.15.
- Val top-1 doğruluk: **0.5156**
- Val loss: 1.2874
- **Torch ↔ numpy parite max fark: 1.9e-06**
- Çıktı: `bot/policy_prior.npz` (173 KB)

Policy prior sadece MAIN kararında rule skoruna küçük bir bonus olarak eklenir.
ATTACH kararlarında ağırlık **0**; Ogerpon/Kangaskhan/Latias/Clefairy gibi
setup PLAY kararlarında ağırlık **0.25×**. Varsayılan `MK_PRIOR_WEIGHT=60`;
`0` ile kapatılabilir.

## Value net

Veri: `python3 agents/mk_ogrepon/tools/build_dataset.py`

- 50 replay, **19.533 public-state örneği** (her iki oyuncu; label maç sonucu).
- 112 boyutlu state.
- Eğitim: `python3 agents/mk_ogrepon/tools/train_value.py`
- Mimari: `112 → 64 → 32 → 1`, ReLU, dropout 0.15.
- 15.641 train / 3.892 val, episode bazlı split.
- Val doğruluk: **0.9825**, val loss 0.0447.
- **Torch ↔ numpy parite max fark: 3.8e-06**
- Çıktı: `bot/value_model.npz` (35 KB)

Not: value net iki oyuncunun da kararlarını ve Top1'in hep aynı desteyi
oynamasını görüyor; yüksek doğruluk kısmen "hangi taraf Top1 destesi" sinyalini
öğrenmesinden geliyor. Bu yüzden ladder'da doğrulanmadan search leaf'ine
ağırlıklı sokulmadı. Varsayılan `MK_VALUE_WEIGHT=0`; `MK_VALUE=0` modeli
tamamen kapatır.

## Dosyalar

- `bot/mk_features.py` — state/option featurizer (state 112, option 192).
- `bot/mk_model.py` — policy prior numpy inference.
- `bot/mk_value.py` — value net numpy inference.
- `bot/policy_prior.npz`, `bot/value_model.npz` — ağırlıklar.
- `tools/train_prior.py`, `tools/train_value.py`, `tools/build_dataset.py`.

Parite testleri eğitim scriptlerinin sonunda otomatik çalışır ve 1e-5'i
aşarsa script hata verir.
