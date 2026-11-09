# Robustness raporu

Komut:

```bash
MK_SEARCH=1 MK_BUDGET=0.18 MK_PRIOR_WEIGHT=60 MK_VALUE_WEIGHT=0 \
  .venv/bin/python agents/mk_ogrepon/tools/stress_test.py 10
```

Rakip seti (her biri 10 maç): `_template`, `banette`, `dragapult`,
`benchmarks/external_agents/alakazam`, `benchmarks/meta_decks/mega_abomasnow`.
50 maç, tekli format, `reverse` almaşık.

## Sonuç

| Metrik | Değer |
|---|---:|
| Maç | 50 |
| Sonuç (P0 / P1) | 30 / 20 |
| **Geçersiz aksiyon** | **0** |
| **Exception** | **0** |
| **Engine error** | **0** |
| **Policy fallback** | **0** |
| Karar sayısı | 2524 |
| Ortalama karar süresi | 1.74 ms |
| p95 karar süresi | 6.32 ms |
| Maksimum karar süresi | 40.6 ms |
| Search override | 253 |
| Search toplam süre | 3.86 s |

Maksimum 40.6 ms, 0.18 s'lik search bütçesinin çok altında. Kaggle'ın
600 s banked overage / 6000 ms runTimeout sınırına göre güvenli.

## Kapsanan seçim bağlamları

`stress_test.py` şu SelectType'ları gördü: `MAIN`, `CARD`, `ATTACHED_CARD`,
`ENERGY`, `EVOLVE`, `COUNT`, `YES_NO`. Bağlamlar:

`MAIN`, `SETUP_ACTIVE`, `SETUP_BENCH`, `SWITCH`, `TO_ACTIVE`, `TO_BENCH`,
`TO_HAND`, `DISCARD`, `TO_DECK`, `DAMAGE_COUNTER_ANY`, `DAMAGE`,
`ATTACH_FROM`, `ATTACH_TO`, `SWITCH_ENERGY_CARD`, `SWITCH_ENERGY`,
`DISCARD_ENERGY`, `TO_DECK_ENERGY`, `DRAW_COUNT`, `IS_FIRST`, `ACTIVATE`,
`FIRST_EFFECT`, `EVOLVE`.

`mk_policy.agent()` her durumda try/except ile legal ilk seçeneğe düşer;
stress testinde fallback'e hiç ihtiyaç olmadı.

## Numpy olmayan ortam

Sistem `python3`'ünde numpy yokken (prior/value devre dışı, search açık)
3 maçlık smoke test de 0 exception / 0 geçersiz aksiyon verdi. Submission
Kaggle'da numpy bulunduğu için prior/value yüklenir; bulunmazsa kural moduna
düşer.

Search kapalı 150 maçlık ek test: 0 geçersiz / 0 exception / 0 fallback,
maksimum 15.1 ms. Makine çıktıları: [`robustness.json`](robustness.json)
(search açık) ve [`robustness_rules.json`](robustness_rules.json) (search kapalı).
