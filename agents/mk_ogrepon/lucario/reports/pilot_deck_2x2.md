# 2x2 — deste mi pilot mu?

200 maç/rakip, seat-swapped, 5 paralel işçi. External pilotta forward search
kapatıldı (sadece politika; aksi halde 1.8 s/karar ile ölçüm pratik değil).

| Pilot + Deste | Dragapult proxy | External Lucario |
|---|---:|---:|
| **Bizim pilot + deck_real_petit** | **30-41.5%** | **20-27%** |
| Bizim pilot + deck_pokepad | 14-17% | 16-18% |
| External pilot + deck_pokepad | **34-35.5%** | 25-28% |
| External pilot + kendi destesi | **54.5-66.5%** | **47-52.5%** |

Aralıklar farklı 200'lük koşuların RNG bandı; her hücre ±~3.5%.

## Hangi eksen ne kadar açıklıyor?

- Aynı bizim pilot: `deck_real_petit` vs `deck_pokepad`
  → Dragapult **+17-24 puan**; deste farkı büyük.
- Aynı `deck_pokepad`: bizim pilot **%14-17**, external pilot **%34-35**
  → pilot farkı **+20 puan**.
- External pilot + kendi destesi: **%54-66** → hem pilot hem deste birlikte.
- External pilot + bizim pokepad: **%34** → iyi pilot zayıf destede de
  bizim pilottan iyi.

Sonuç: **iki eksen de önemli**. Pilot açığı yaklaşık 20 puan, deste açığı
yaklaşık 17-24 puan. Bu yüzden sadece ağırlık ayarı değil, karar mantığı +
tutarlı deste birlikte gerekli.

## Mirror gate

Yeni gate hedefi mirror ≥ %40. En iyi bizim pilot + deck_real_petit + prior
koşusu **%26** civarında; **geçmiyor**. External own mirror **%47-52**.
