# Kart davranışı doğrulaması — MK / Ogerpon

Kaynak: `pokemon/ptcg_engine_playground/ptcgProgram 22/CardImpl.h` ve
`replays/top5_nov1/17014244` içindeki 50 Top1 galibiyet replay'i.
Makine çıktısı: [`card_behavior.json`](card_behavior.json).

## Deste doğrulaması

- `deck_top1.csv` / `bot/deck.csv`: **60 kart**, 23 farklı kart.
- **17 Basic Pokémon kopyası**, 10 farklı Basic Pokémon satırı.
- **1 ACE SPEC**: Unfair Stamp (1080).
- **Tera**: Teal Mask Ogerpon ex (96), Wellspring Mask Ogerpon ex (108).
- **Mega ex**: Mega Kangaskhan ex (756) → `megaEx=True`, engine prize değeri **3**.
- Radiant yok.
- `python3 tools/check_deck.py agents/mk_ogrepon/deck_top1.csv --engine` → **deck accepted**.

Top1 listesi 1B taslağından farklı çıktı. Gerçek liste:

| Kart | Adet | Kart | Adet |
|---|---:|---|---:|
| Basic {G} Energy (1) | 11 | Prism Energy (16) | 4 |
| Teal Mask Ogerpon ex (96) | 4 | Wellspring Mask Ogerpon ex (108) | 1 |
| Mega Kangaskhan ex (756) | 2 | Latias ex (184) | 2 |
| Lillie's Clefairy ex (272) | 2 | Meowth ex (1071) | 2 |
| Fezandipiti ex (140) | 1 | Munkidori (112) | 1 |
| Pecharunt (230) | 1 | Chi-Yu (31) | 1 |
| Energy Switch (1116) | 4 | Ultra Ball (1121) | 4 |
| Bug Catching Set (1094) | 3 | Boss's Orders (1182) | 3 |
| Lillie's Determination (1227) | 4 | Area Zero Underdepths (1250) | 3 |
| Night Stretcher (1097) | 2 | Tera Orb (1127) | 2 |
| Unfair Stamp (1080) | 1 | Ciphermaniac's Codebreaking (1188) | 1 |
| N's Plan (1221) | 1 | | |

**1B taslağı düzeltmeleri**

- 1B'deki “Mega Latias ex” gerçek listede yok; kart **Latias ex (184)**.
- Poké Pad, Pokégear 3.0, Switch, Judge, Prime Catcher ve Energy Retrieval
  gerçek Top1 listesinde yok.
- Gerçek listede Wellspring Ogerpon, Bug Catching Set, Tera Orb,
  Ciphermaniac, N's Plan ve Unfair Stamp var.

## Doğrulanan kart davranışları

| Kart | Kaynak davranışı | Replay/sim doğrulaması |
|---|---|---|
| Teal Mask Ogerpon ex (96) | `Teal Dance`: elden **Basic {G} Energy** kartını bu Pokémon'a tak, sonra 1 çek. Saldırı `Myriad Leaf Shower`: 30 + iki aktifteki enerji sayısı × 30, GGG. Tera. | 676 kullanım; **676/676** kullanımda seçilen Ogerpon'un enerjisi **+1**, deste **-1** (iki adım sonra ölçüldü). |
| Mega Kangaskhan ex (756) | `Run Errand`: aktifken turda 1 kez, 2 çek. `Rapid-Fire Combo`: 200 + yazı-tura gelene kadar her tura 50, CCC. Mega ex → 3 prize. | 306 kullanım; **306/306** kullanımda deste **-2**, el **+2**. Prize değeri engine'de **3**. |
| Fezandipiti ex (140) | `Flip the Script`: önceki rakip turunda Pokémon'un KO olduysa 3 çek. `Cruel Arrow`: herhangi bir rakibe 100. | 75 kullanım; **75/75** kullanımda deste **-3**. |
| Lillie's Determination (1227) | Eli desteye karıştır; 6 çek. Tam **6 prize** kaldıysa 8 çek. | 208 kullanım. 6 prize: **8 çek** (120/120). Diğer prize sayıları: **6 çek**. |
| Area Zero Underdepths (1250) | Tera Pokémon'u olan oyuncunun bench limiti 8. Tera yoksa 5. | 142 kullanım. Bizde Tera varken **5→8** (132 durum). Rakip Tera yoksa 5'te kaldı (85 durum); rakipte de Tera varsa 5→8 (47 durum). |
| Prism Energy (16) | Basic Pokémon'a takılıysa **her türden 1 enerji**; evrimde yalnız Colorless 1. | Basic Pokémon'larda enerji tipi listesinde **10 (Rainbow)** görüldü: `(10,)`, `(1,10)`, `(1,1,10)` vb. Bu destede tüm Pokémon Basic olduğu için Prism pratikte her rengi öder. |
| Energy Switch (1116) | Yalnız **Basic Energy**'yi bir Pokémon'dan diğerine taşır. | 213 Top1 kullanımı; bizim destede taşınabilen tek Basic enerji **Basic {G}**. Prism taşınamaz. |
| N's Plan (1221) | Bench'teki Pokémon'lardan **2 enerjiye kadar** aktif Pokémon'a taşır. | Kaynak metni ve sim seçimleri `SWITCH_ENERGY` bağlamını doğruluyor. Prism dahil her enerji taşınabilir. |
| Boss's Orders (1182) | Rakip bench'inden 1 Pokémon'u aktif yapar. | Replay'lerde `SWITCH` bağlamında hedef bench seçimi doğrulandı. |
| Unfair Stamp (1080) | Önceki rakip turunda KO olduysa kullanılır; iki el desteye karışır, biz 5, rakip 2 çeker. ACE SPEC. | `CardImpl` koşulu ve engine `ACE SPEC` sayısı doğrulandı. |
| Bug Catching Set (1094) | Üst 7 karttan en fazla 2 **Grass Pokémon / Basic {G} Energy** al. | `TO_HAND` seçimleri `area=12 (LOOKING)` üzerinden doğrulandı. |
| Tera Orb (1127) | Desteden 1 **Tera** Pokémon arar. | Ogerpon aramalarında kullanıldı; Ogerpon Tera=True. |
| Ciphermaniac's Codebreaking (1188) | Desteden 2 kart seçer, desteyi karıştırır, seçilenleri istediğin sırayla üste koyar. | Kaynak metni; `TO_DECK` seçim bağlamı. |
| Night Stretcher (1097) | Discard'dan **Pokémon veya Basic Energy** alır. | Prism **Special Energy** olduğu için Night Stretcher ile alınamaz; kural motoru bunu `BASIC_ENERGY` filtresiyle uygular. |
| Ultra Ball (1121) | 2 kart discard et, desteden Pokémon ara. | 222 Top1 kullanımı; discard seçimleri `DISCARD` bağlamında. |
| Meowth ex (1071) | Bench'e oynanınca Supporter arar. `Tuck Tail`: 60 ve kendini/eklentilerini ele döndürür. | `ACTIVATE` + `TO_HAND` supporter seçimi doğrulandı. |
| Latias ex (184) | `Skyliner`: Basic Pokémon'ların retreat maliyeti 0. `Eon Blade`: 200, sonraki tur saldıramaz. | Pasif yetenek; top1 listesinde 2 adet. |
| Lillie's Clefairy ex (272) | `Fairy Zone`: rakip Dragon Pokémon'ların weakness'ı Psychic. `Full Moon Rondo`: 20 + iki benchteki Pokémon sayısı × 20. | Dragon karşılaşmalarında `Clefairy` deste planının anahtarı. |
| Munkidori (112) | Darkness enerjisi varsa, kendi hasarlı Pokémon'undan 3 damage counter'ı rakibe taşır. | Prism Basic'te Darkness dahil her türü sağlar; yetenek koşulu buna bağlı. |
| Pecharunt (230) | Aktifken rakibin zehirli Pokémon'u checkup'ta +5 counter (50 hasar) alır. `Poison Chain`: 10 + zehir + retreat kilidi. | Kaynak metni; zehir planı yüksek HP tanklara karşı. |
| Chi-Yu (31) | `Allure`: 2 çek. `Ground Melter`: 60, stadium varsa +60 ve stadium'u discard eder. | Kaynak metni; Area Zero'yu yakar, dikkat. |
| Wellspring Ogerpon ex (108) | `Sob`: 20, rakip retreat edemez. `Torrential Pump`: 100; isterse 3 enerjiyi desteye döndürüp bir bench'e 120. | Kaynak metni; Water enerji için Prism gerekir. |

## Saldırı kullanım dağılımı (50 Top1 galibiyeti)

| Saldırı | Kullanım |
|---|---:|
| Myriad Leaf Shower (96) | 112 |
| Sob (108) | 105 |
| Rapid-Fire Combo (756) | 96 |
| Full Moon Rondo (272) | 78 |
| Poison Chain (230) | 59 |
| Torrential Pump (108) | 42 |
| Allure (31) | 20 |
| Cruel Arrow (140) | 15 |
| Ground Melter (31) | 9 |
| Mind Bend (112) | 6 |
| Eon Blade (184) | 1 |

## Notlar / riskler

- `Tera` kuralı: bench'teki Tera Pokémon saldırılardan hasar almaz; kural
  motoru bench hedeflerinde bunu 0 hasar sayar.
- Mega Kangaskhan ex 3 prize verir. Kural motoru `prize_value` ile bunu
  hesaplar; rakip 3 prize kala Mega'yı öne sürme cezası uygular.
- Area Zero, Tera sahada değilken bench'i 8 yapmaz; motor `benchMax` alanını
  kullanır.
- Prism Energy evrim Pokémon'unda yalnız Colorless olur; bu destede evrim yok.
- 1B'deki “Poké Pad / Prime Catcher / Judge / Switch” bu destede yok; kart
  metnini hafızadan varsaymak yerine gerçek Top1 listesi kullanıldı.
