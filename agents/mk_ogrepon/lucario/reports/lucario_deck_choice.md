# Deste kararı — 4 varyant

Her varyant, 8 rakipli havuzda **rakip başına 200 maç**, seat-swapped
(yarı P0, yarı P1) oynandı. Not: engine seed'i olmadığı için bu karşılaştırma
fork-paired değil; aynı rakip/deck sayısı ve koltuk dengesi kullanıldı.

| Varyant | 7-rakip ort. | 7-rakip min | En kötü | Not |
|---|---:|---:|---|---|
| `deck.csv` (677 + Poffin) | **35.5%** | 10.0% | Dragapult 10% | Poffin yalnız Budew alıyor |
| `deck_riolu974.csv` (974 + Poffin) | 34.4% | 7.0% | Dragapult 7% | Poffin Riolu alıyor |
| `deck_pokepad.csv` (677 + Poké Pad) | 33.5% | **11.0%** | Alakazam 11% | Poffin sorunu yok |
| `deck_974_pokepad.csv` (974 + Poké Pad) | 33.0% | 9.5% | Dragapult 9.5% | İki düzeltme birlikte |

Tam 8-rakip ortalamaları (noisy Dragapult dahil): %32.4 / %31.9 / %31.6 /
%30.8.

## Anlamlılık

200×7 = 1400 maç için ortalama farkların yaklaşık SE'si ~1.2-1.3%.
En büyük fark (deck.csv - 974_pokepad) ~2.5% → yaklaşık 2 SE; tek başına
anlamlı kabul edilmedi. Kullanıcı talimatındaki "anlamlı değilse
`deck_974_pokepad`" kuralı gereği aday **974_pokepad** tutuldu.

Ancak submission gate'i açısından 974_pokepad, baseline `deck.csv`'den
ortalama ve en kötü eşleşmede düşük; bu yüzden **submit'e hazır değil**.

## Poffin doğrulaması

`deck_riolu974.csv` ile 20 oyunluk log taramasında Poffin seçimleri:

- `context=5 (TO_BENCH)`, chosen: **Riolu 974 + Riolu 974**
- birçok oyunda tek/çift **Riolu 974**
- `deck.csv` (677) ile aynı tarama: Poffin -> yalnız **Budew 235**

Ham log: `reports/poffin_verification.json`.

## Sonuç

- Poffin + Riolu 677 **gerçekten uyumsuz**; 4 kart ölü.
- 974 + Poffin çalışıyor; fakat Riolu 974'ün saldırısı {F}{C}, 677'nin {F}.
  Bu yüzden 974 varyantı tek enerjiyle saldıramıyor.
- 677 + Poké Pad, Poffin sorununu çözüyor ve en iyi minimumu veriyor
  (%11); ortalama farkı anlamlı değil.
- Öneri: **677 + Poké Pad** (`deck_pokepad`) veya gerçek Petit Canard listesi;
  974_pokepad yalnızca kullanıcı fallback kuralı nedeniyle aday.

## Dosyalar

`reports/final_200_*.json` — tüm rakip bazlı sonuçlar.
