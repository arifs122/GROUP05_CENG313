# Rakip havuzu

`tools/eval_pool.py` havuzu:

1. `bots/_template` — sample Mega Abomasnow
2. `bots/dragapult` — 2nd-place Dragapult **proxy** (public sample policy;
   gerçek replay klonu değil)
3. `benchmarks/external_agents/alakazam`
4. `bots/banette`
5. `benchmarks/external_agents/lucario_mega_hariyama` — external mirror
6. `benchmarks/external_agents/bellibolt`
7. `benchmarks/meta_decks/mega_abomasnow`
8. `noisy:bots/dragapult` — her kararda **%7 rastgele legal hamle**

Not: 770 replay'den eğitilmiş gerçek bir Dragapult davranış klonu bu turda
eğitilmedi; mevcut `bots/dragapult` ve gürültülü varyantı proxy olarak
kullanıldı. Bu raporda açıkça belirtilmelidir.

## 200 maçlık seçili deck (`deck_974_pokepad`)

| Rakip | W-L | Win |
|---|---:|---:|
| template | 86-114 | 43.0% |
| dragapult | 19-181 | **9.5%** |
| alakazam | 23-177 | 11.5% |
| banette | 92-108 | 46.0% |
| lucario_mega_hariyama | 19-181 | 9.5% |
| bellibolt | 121-78 | 60.8% |
| mega_abomasnow | 102-98 | 51.0% |
| noisy_dragapult | 30-170 | 15.0% |

- 7-rakip ortalaması: **33.0%**
- En kötü eşleşme: **Dragapult %9.5**
- Gürültülü Dragapult %15.0; gürültü bu küçük örnekte daha kolay göründü,
  ama tek başına anlamlı değil.

## Ana metrik

Ortalama değil, **en kötü eşleşme** kullanılmalı: Dragapult/Alakazam ~%10.
Bu, submission gate'ini geçmiyor.
