> **Not:** Bu dosya ilk 20 maçlık taramadır. Güncel ölçümler için
> `measurement.md`, `lucario_deck_choice.md`, `opponents.md`.

# Lucario — lokal rakip havuzu ve ağırlık denemesi

## Rakip havuzu

`agents/mk_ogrepon/lucario/tools/eval_pool.py` şu public botları kullanır:

- `bots/_template` — sample Mega Abomasnow
- `bots/dragapult` — Dragapult spread
- `benchmarks/external_agents/alakazam`
- `bots/banette`
- `benchmarks/external_agents/lucario_mega_hariyama` — Mega Lucario mirror
- `benchmarks/external_agents/bellibolt`
- `benchmarks/meta_decks/mega_abomasnow`

Kullanım:

```bash
PTCG_BELIEF_LOG=0 python3 agents/mk_ogrepon/lucario/tools/eval_pool.py \
  --games 20 --deck agents/mk_ogrepon/lucario/deck.csv
```

## Kullanıcı destesi — 20 maç/rakip

| Rakip | W-L | Win rate |
|---|---:|---:|
| template | 9-11 | 45% |
| dragapult | 1-19 | **5%** |
| alakazam | 4-16 | 20% |
| banette | 10-10 | 50% |
| lucario_mega_hariyama | 2-18 | **10%** |
| bellibolt | 12-8 | 60% |
| mega_abomasnow | 11-9 | 55% |
| **ortalama** | | **35%** |
| **en kötü** | | **5%** |

Ham çıktı: `reports/pool_baseline.json`.

## Heuristic düzeltmelerinden sonra

- `Switch` yalnız aktif attacker değilken değil; Mega bench'te enerji varsa da
  kullanılabilir hale getirildi.
- `Retreat` Mega bench'te hazırsa açıldı.
- `Hariyama` evolve ability'si Boss seviyesine yakın puanlandı.
- `Boss’s Orders` / `Hero’s Cape` isimlerindeki curly apostrophe eşleşmesi
  düzeltildi.
- `O_ATTACH` tool kartları artık `item` olarak sınıflanıyor; Hero's Cape yanlış
  Pokémon'a takılmıyor.
- `Poffin` hedefi desteye göre kontrol ediliyor.

Yeni 20 maçlık havuz (aynı gün, farklı RNG):

```
template 50%, dragapult 10%, alakazam 0%, banette 60%,
mirror 0%, bellibolt 70%, mega_abomasnow 50%
mean 34%, min 0%
```

Bu sayılar 20 maçla çok gürültülü; **karar kriteri değildir**. 100 maçlık
template testleri 34-51% arasında değişti.

## Deste varyantları (100 maç, template)

| Varyant | P0 win |
|---|---:|
| Kullanıcı: 677 + Poffin | 34-51% (seed'e göre) |
| 677 → 974 (Poffin çalışır) | 40-46% |
| Poffin → Poké Pad (677 kalır) | 39-47% |
| Gerçek Petit Canard listesi | 34-47% |

Deste farkı tek başına net değil; asıl sorun Poffin/677 uyumsuzluğu ve
Alakazam/Dragapult gibi zor eşleşmeler.

## CMA-ES denemesi

`tools/cma_es.py` ile 20 ağırlık üzerinde küçük bir CMA-ES çalıştırıldı:

- 6 generation, popülasyon 10, train havuzunda 8 maç/rakip
- Train: template, banette, mega_abomasnow, bellibolt
- Holdout: dragapult, alakazam, lucario mirror

En iyi aday train'de 0.50 min / 0.66 mean gösterdi, fakat aynı ağırlıkların
yeniden ölçümü train 0.125 / holdout 0.10 verdi. Bu, 8 maçlık değerlendirmenin
aşırı gürültülü olduğunu gösteriyor; **`weights.json` olarak kullanılmadı**.
Rapor: `reports/cma_es_user.json`.

Sonraki adım için doğru yaklaşım: aynı rakip havuzunda 100+ maç/aday, daha
düşük boyutlu (6-8) ağırlık seti ve ayrı holdout; engine deal seed'i
olmadığından gürültüyü azaltmak için tekrar sayısını artırmak gerekiyor.

## Değerlendirme komutu

```bash
PTCG_BELIEF_LOG=0 python3 agents/mk_ogrepon/lucario/tools/eval_pool.py \
  --games 20 --deck agents/mk_ogrepon/lucario/deck.csv \
  --out agents/mk_ogrepon/lucario/reports/pool_baseline.json
```
