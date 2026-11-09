# Kayıp maç analizi — 40 kayıp

Seçili aday deste: `deck_974_pokepad.csv`.
`tools/analyze_losses.py` 64 maçta 40 kayıp topladı; her kayıpta ilk
"muhtemel hata" etiketlendi.

## Etiket dağılımı (son heuristic)

| Etiket | Sayı |
|---|---:|
| other | 30 |
| delayed_mega | 3 |
| late_bench | 3 |
| wrong_energy | 3 |
| search_spam | 1 |

`other`, tek bir kararla açıklanamayan kayıplar: Mega Lucario'nun hiç
çekilmemesi / rakip baskısı / deste dağılımı. Bu bir aksiyon hatası değil.

`other` 30 kayıp çoğunlukla Mega Lucario'nun hiç çekilmemesi / rakip
baskısı; tek bir karar hatası değil.

## Aksiyon kuralına çevrilen hatalar

### 1. Hero's Cape yanlış hedef
Eski analizde 6-9 kayıpta Cape, Lunatone/Makuhita gibi hedeflere takılıyordu.
Kural:
- `Hero's Cape` yalnızca **Mega Lucario ex** veya **Hariyama** hedefinde
  pozitif skor alır; diğer hedeflerde **-10** (END 0'dan düşük).
- Regresyon: `tests/test_regressions.py::test_cape_only_on_mega_or_hariyama`.

### 2. Poffin / Riolu 677 uyumsuzluğu
Kural:
- Poffin skoru deste-aware: HP≤70 hedefi kalmadıysa -10.
- `deck_riolu974` ve `deck_pokepad` varyantları eklendi.
- Regresyon: `test_poffin_deck_awareness`.

### 3. Yanlış enerji hedefi / geç gelişim
Kural:
- Enerji, attacker olmayan Pokémon'da 200 taban; Riolu/Makuhita/Hariyama/
  Mega'da 550+.
- Mega bench'te enerji varsa Switch/Retreat açıldı.
- Hariyama evolve ability'si Boss seviyesine yakın puanlandı.
- Regresyon: `test_attach_to_attacker_beats_non_attacker`,
  `test_retreat_without_charged_bench_is_penalized`,
  `test_boss_without_setup_or_ko_is_not_played`.

## Diğer kurallar

- `Boss's Orders`: hedef setup Pokémon'u değilse ve KO yoksa -200.
- `Premium Power Pro`: aktif attacker değilse -10.
- `Switch`: aktif attacker ve bench'te enerjili Mega yoksa -100.
- Deterministik tie-break (`choose` rastgele değil) → ölçüm tekrarlanabilir.

## Regresyon testleri

```bash
python3 -m unittest agents/mk_ogrepon/lucario/tests/test_regressions.py -v
# 7 test OK
```

Testler saf `State`/`Act` üzerinde; motor gerektirmez.
