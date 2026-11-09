# Ladder replay deck-vs-deck matrisi

Kaynak: `replays/**/*.json`, EpisodeId ile dedupe; **1920 episode**.
Archetype sınıflandırması `tools/replay_meta.py` üzerinden.
Makine çıktısı: `reports/deck_matrix.json`.

## Deste payı

Dragapult archetype'ı tüm deck görünümlerinde **%52.5** paya sahip.
Bu, ladder'da Dragapult eşleşmesinin ne kadar kritik olduğunu doğruluyor.

Ladder'da Dragapult'a karşı en iyi görünenler:

| Dragapult'u yenen | Win rate | Maç |
|---|---:|---:|
| Mega Lucario | **62.5%** *(Mega Lucario P0 iken)* | 32 |
| Mega Kangaskhan | 55.3% | 188 |
| Banette | 51.7% | 29 |
| Mega Starmie | 63.6% | 22 |
| Slowking-Kangaskhan | 72.7% | 11 |
| Dwebble | 93.8% | 16 |

Ters yön (Dragapult P0 iken Mega Lucario'ya karşı) **42.9%** (24/56).
İki yön birleşince **Mega Lucario vs Dragapult ≈ 50% (44/88)**.
Yani Mega Lucario, Dragapult'a karşı ladder'da dengeli; lokal %9.5-13.5
politika/deck uyumsuzluğunun göstergesi.

## Dragapult'un zorlandığı eşleşmeler

- Mega Lucario ~50%
- Mega Kangaskhan 55.3% (Dragapult karşısında Kangaskhan lehine)
- Banette 51.7% Banette lehine
- Alakazam Dragapult'a karşı sadece **19.3%** (Dragapult 62.3% favori)
- Mega Abomasnow Dragapult'a karşı 34.8%; Dragapult 67.7% favori
- Mega Excadrill Dragapult'a karşı 33.3%; Dragapult 66.7% favori

## Lucario için anlam

- Ladder'da Dragapult payı yüksek (%52.5) → Dragapult eşleşmesi kaçınılmaz.
- Ladder Mega Lucario Dragapult'a ~%50; lokal bot %9.5-13.5. Fark politika
  ve deste tutarlılığında.
- Alakazam, Mega Lucario'nun en kötü ladder eşleşmesi (~%19 Mega Lucario
  lehine); lokal Alakazam da %10-16. Bu gerçek bir counter.
- Poffin/Riolu uyumsuzluğu giderilmeden Dragapult oranı yükseltilemez.
