# Submission / aşama notları

Bu Çalışmada **hiçbir submission Kaggle'a yüklenmedi**. Aşağıdaki sıra,
ladder'da okuma yapılacaksa kullanılacak öneridir.

## Ortak paket

```bash
.venv/bin/python tools/build_submission.py --bot agents/mk_ogrepon/bot
```

Kök: `main.py`, `deck.csv`, `mk_*.py`, `policy_prior.npz`,
`value_model.npz`, `cg/`. Boyut ~2.25 MiB (limit ~197.7 MiB).

## Aşamalar

| Aşama | Env | Ne değişti | Neye karşı okunur |
|---|---|---|---|
| A0 kural | `MK_SEARCH=0 MK_PRIOR=0 MK_VALUE=0` | Katman 0/1, legal kural çekirdeği | İlk baseline; ladder başlangıcı |
| A1 search | `MK_SEARCH=1 MK_BUDGET=0.18 MK_PRIOR=0` | Determinised 1-ply search | A0'a karşı |
| A2 + prior | `MK_PRIOR=1 MK_PRIOR_WEIGHT=60` | Top1 policy prior bonus | A1'e karşı |
| A3 + value | `MK_VALUE=1 MK_VALUE_WEIGHT=0.2` | Value net leaf katkısı | A2'ye karşı |

A3 varsayılan olarak `MK_VALUE_WEIGHT=0` gelir. Value net lokal A/B'de
(20 maçlık, 0.3/0.8 ağırlık) zarar verdi; ladder'da net artış görülmezse
kapalı kalmalı.

## Notlar

- Her aşama ayrı submission olarak okunmalı; aynı anda 2+ deney açmamak
  için önceki iyi sürüm ezilmemeli.
- 20-30 BO3'lik lokal testler karar için yeterli değildir; crash/legal
  kontrolü için kullanıldı.
- En az 500 ladder maçı olmadan ağırlık değiştirmemek gerekir.
- Geri dönüş: `MK_SEARCH=0`, `MK_PRIOR=0`, `MK_VALUE=0` ile saf kural
  moduna dönülür; bu mod stress testinde 0 hata verdi.

## Bu çalışmadaki lokal A/B özeti

- Aynı 5 rakip × 30 maç seed 88: prior 0 → 77/150; prior 60 → 87/150;
  prior 120 → 75/150; prior 220 → 79/150. Bu yüzden varsayılan 60.
- Aynı 5 rakip × 20 maç seed 91: search kapalı 49/100, search açık 52/100.
- Value ağırlığı 0.3/0.8, 20 maçlık testte sırasıyla 9/20 ve 7/20;
  value 0'da 11/20. Bu yüzden value varsayılan kapalı.
