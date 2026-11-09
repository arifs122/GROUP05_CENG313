# Karar ayrışma analizi — Mega Lucario MAIN

Kaynak: `replays/**/*.json` içindeki **114 Mega Lucario episode** (678),
toplam **9.246 MAIN kararı**. Replay'deki güçlü pilotun seçtiği aksiyon
etiket; bizim `to_state`/`to_acts` + skor + policy prior seçimimiz tahmin.

## Top-1 uyum

| Pilot | Top-1 |
|---|---:|
| Bizim heuristik + BC prior | **58.4%** |
| Bizim saf heuristik (prior yok) | 40.6% |
| External Lucario referansı (search kapalı) | **66.9%** |

Hedef 80% **tutmadı**; prior farkı 33.4 → 40.6 → 58.4'e çıkardı.

## Karar tipine göre

| Kategori | Bizim | External |
|---|---:|---:|
| ability | 87.5% | 97.4% |
| evolve | 78.0% | 62.3% |
| boss | 55.0% | 43.8% |
| search | 68.0% | 74.9% |
| basic | 62.7% | 83.1% |
| energy | 41.1% | 47.7% |
| attack | 51.6% | 52.9% |
| retreat | **5.1%** | 26.1% |
| end | 46.9% | 73.7% |
| other | 48.8% | 64.5% |

## En sık 10 ayrışma (expert → biz)

| # | Adet | Kategori | Expert | Biz |
|---:|---:|---|---|---|
| 1 | 93 | ability | Lunatone ability | Evolve Mega |
| 2 | 82 | energy | Attach F → Mega | Lunatone ability |
| 3 | 80 | other | Premium Power Pro | Boss’s Orders |
| 4 | 74 | search | Fighting Gong | Attach F → Riolu |
| 5 | 74 | other | Lillie's Determination | Boss’s Orders |
| 6 | 74 | energy | Attach F → Mega | Boss’s Orders |
| 7 | 59 | energy | Attach F → Solrock | Lunatone ability |
| 8 | 58 | attack | Aura Jab | Mega Brave |
| 9 | 56 | search | Dusk Ball | Attach F → Riolu |
| 10 | 53 | basic | Play Lunatone | Attach F → Riolu |

## Kurala çevrilenler

- **Solrock/Lunatone bench önceliği** (basic skoru 680; bench 5'e kadar).
- **Solrock/Lunatone enerji hedefi** (attach base 470; sadece Mega/Riolu değil).
- **Lunatone ability 980**; artık saldırıdan önce kullanılıyor.
- **Aura Jab erken +120**, Mega Brave erken -80; expert Aura Jab'ı tercih ediyor.
- **Dusk Ball / Poké Pad / Fighting Gong** arama skoru 720/730; attach'tan önce.
- **Carmine 760** (external deckte var).
- **Hariyama ability** Boss seviyesinde.
- **Deterministik tie-break** (paired ölçüm için).

Her kural için `tests/test_regressions.py` altında test var (7 test).

## BC prior

`tools/train_prior.py`:
- 59 galibiyet episode, 4.851 MAIN örneği.
- State 112 + option 192 → 128 → 64 → 1.
- Val top-1 **46.6%**, Torch↔numpy parite **1.9e-6**.
- Runtime'da ana karar `score + 260 × (logit - mean)`, game-winning lethal
  kuralı korunuyor.

Prior'lı uyum 58.4%; 80% hedefi için daha fazla/güçlü etiket veya daha iyi
state temsili gerekiyor.
