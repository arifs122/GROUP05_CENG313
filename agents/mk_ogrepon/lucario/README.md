# Mega Lucario / Hariyama ajanı

Bu klasör, kullanıcıdan gelen `lucario_agent.py` heuristiğinin motor adaptörü
ve yerel test ortamıdır. **Henüz Kaggle'a submit edilmedi.**

## Dosyalar

| Dosya | Görev |
|---|---|
| `main.py` | Kaggle/arena giriş noktası |
| `lucario_agent.py` | Heuristik + `to_state` / `to_acts` adaptörleri |
| `mk_state.py` | Gözlem yardımcıları |
| `mk_data.py` | Kart metadata yardımcıları |
| `deck.csv` | Kullanıcı destesi (60 kart) |
| `deck_riolu974.csv` | Poffin'in çalıştığı Riolu 974 varyantı |
| `deck_pokepad.csv` | Poffin → Poké Pad varyantı |
| `deck_real_petit.csv` | Replay'den Petit Canard listesi |
| `tools/eval_pool.py` | Rakip havuzu değerlendirmesi |
| `tools/cma_es.py` | Ağırlık CMA-ES denemesi |
| `reports/` | Kart doğrulama, havuz ve deste raporları |

## Çalıştırma

```bash
# tek maç
python3 tools/arena.py \
  --bot0 agents/mk_ogrepon/lucario \
  --bot1 bots/_template --games 20 --seed 7

# BO3
python3 tools/arena.py \
  --bot0 agents/mk_ogrepon/lucario \
  --bot1 bots/dragapult --games 10 --bo3 --seed 7

# rakip havuzu
PTCG_BELIEF_LOG=0 python3 agents/mk_ogrepon/lucario/tools/eval_pool.py \
  --games 20 --deck agents/mk_ogrepon/lucario/deck.csv
```

## Adaptör notları

- `to_state`: public gözlem + kendi el sayısı; rakip el/prize gizli bilgisi yok.
- `to_acts`: MAIN için `Act(kind, card, target, ko, prize)` üretir; saldırı
  hasarı weakness/resistance ile hesaplanır.
- Tool `ATTACH` seçenekleri `item` olarak sınıflandırılır (Hero's Cape).
- Çoklu seçimlerde (`Poffin`, Ultra Ball discard, setup bench) `_pick_multi`
  kullanılır; discard'te en düşük değerli kart seçilir.
- Her hata durumunda legal ilk hamleye düşer.

## Önemli bulgu

Kullanıcı destesindeki **Poffin + Riolu 677** uyumsuz: Poffin HP ≤70 arıyor,
Riolu 677 HP 80. Bu yüzden 4 Poffin pratikte sadece Budew'i bulabilir.
Düzeltme seçenekleri:
- Poffin kalsın → Riolu 677 yerine 974/333 (HP70)
- Riolu 677 kalsın → Poffin yerine Poké Pad/Dusk Ball

Ayrıntı: [`reports/lucario_card_behavior.md`](reports/lucario_card_behavior.md)

## Ölçüm düzeni

- `tools/paired_arena.py`: `BattleStart` seed'i olmadığı için bir deal başlatıp
  `fork()` ile adayları klonlar; `GameConfig.deviceRand` kapatılarak coin RNG
  de kopyalanır. Policy/ağırlık testleri paired.
- `tools/eval_pool.py`: deste varyantları için seat-swapped 200 maç/rakip.
- Self-mirror **53-46-1** (~%50); external Lucario aynı desteyle %74 → sorun
  koltuk/bug değil, external politika daha güçlü.
- Ladder replay matrisi: Dragapult payı **%52.5**, Mega Lucario vs Dragapult
  iki yön toplamı ~%50.

## 2x2 — deste/pilot

| Pilot + Deste | Dragapult proxy | External Lucario |
|---|---:|---:|
| Bizim pilot + deck_real_petit | **30-41.5%** | 20-27% |
| Bizim pilot + deck_pokepad | 14-17% | 16-18% |
| External pilot + deck_pokepad | 34-35.5% | 25-28% |
| External pilot + kendi destesi | **54.5-66.5%** | 47-52.5% |

Pilot açığı ~20 puan, deste açığı ~17-24 puan. BC prior MAIN uyumu
**%58.4**; external **%66.9**; hedef %80.

**Yeni gate FAIL:** Dragapult %30+ geçiyor, mirror %20-27 (<%40).
`weights.json` yazılmadı, submit yapılmadı.

Raporlar:
- [`reports/measurement.md`](reports/measurement.md)
- [`reports/lucario_deck_choice.md`](reports/lucario_deck_choice.md)
- [`reports/loss_analysis.md`](reports/loss_analysis.md)
- [`reports/opponents.md`](reports/opponents.md)
- [`reports/deck_matrix.md`](reports/deck_matrix.md)
- [`reports/pilot_deck_2x2.md`](reports/pilot_deck_2x2.md)
- [`reports/decision_divergence.md`](reports/decision_divergence.md)
- [`reports/dragapult_plan.md`](reports/dragapult_plan.md)
- [`reports/submission_gate.md`](reports/submission_gate.md)
- [`reports/lucario_card_behavior.md`](reports/lucario_card_behavior.md)
- [`reports/lucario_robustness.md`](reports/lucario_robustness.md)
