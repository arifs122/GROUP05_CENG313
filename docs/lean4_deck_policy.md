# Lean4 Dragapult — Deck ve Rule Policy

> Bu, **lean4 kendi deck’i**dir. Top2/Petit Canard deck’i değildir.
> Base bot: `bots/dragapult_lean4`
> AI’sız/search kapalı sürüm: `bots/dragapult_lean4_noai`

## Deck (60 kart)

| Adet | Kart ID | Kart |
|---:|---:|---|
| 5 | 2 | Basic Fire Energy |
| 5 | 5 | Basic Psychic Energy |
| 4 | 119 | Dreepy |
| 4 | 120 | Drakloak |
| 3 | 121 | Dragapult ex |
| 3 | 235 | Budew |
| 1 | 140 | Fezandipiti ex |
| 1 | 1071 | Meowth ex |
| 1 | 1079 | Rare Candy |
| 1 | 1080 | Unfair Stamp |
| 4 | 1086 | Buddy-Buddy Poffin |
| 4 | 1120 | Crushing Hammer |
| 4 | 1121 | Ultra Ball |
| 4 | 1152 | Poké Pad |
| 4 | 1227 | Lillie's Determination |
| 3 | 1182 | Boss's Orders |
| 3 | 1198 | Crispin |
| 2 | 1097 | Night Stretcher |
| 2 | 1213 | Judge |
| 2 | 1256 | Team Rocket's Watchtower |

Toplam: **60**

### Lean4’te olmayan önemli kartlar

- **Munkidori yok**
- **Chi-Yu yok**
- **Latias ex yok**
- **2. Rare Candy yok**
- **Dusknoir yok**

Bu yüzden Top2 deck’ine lean4 pilotu takınca Munkidori/Chi-Yu/Crispin hattı ve setup planı uyuşmuyor.

## Policy’nin ana sırası

Kaynak: `bots/dragapult_lean4/main.py`, `_policy()`.

1. **Deck seçimi**
   - `select is None` ise 60 kartlık desteyi döndürür.

2. **Ana karar (MAIN)**
   - Her legal option’a bir skor verir.
   - Skorlar kabaca:
     - evolve / setup
     - ability
     - enerji attach
     - item / supporter play
     - attack
     - retreat
     - end
   - En yüksek skorlu `maxCount` kadar aksiyonu seçer.
   - Skorların hepsi negatifse minimum sayıyı doldurur; asla illegal aksiyon döndürmez.

3. **Setup / gelişim**
   - Dreepy → Drakloak → Dragapult ex hattı.
   - `Rare Candy + Dragapult ex` varsa hızlı evrim.
   - `Buddy-Buddy Poffin`, `Ultra Ball`, `Poké Pad` ile setup arar.
   - `Night Stretcher` ile discard’taki Pokémon/enerjiyi geri alır.

4. **Enerji**
   - Dragapult ex için normalde **2 enerji**.
   - Rakip **Nighttime Mine** oynarsa Tera vergisi nedeniyle **3 enerji**.
   - Fire/Psychic enerjiyi doğru hedefe bağlar.
   - `Crushing Hammer` ile rakip enerjisini bozmaya çalışır.

5. **Saldırı**
   - `main_option_proc` ile saldırı planı çıkarır.
   - Dragapult ex’in **Phantom Dive** saldırısı ana plan.
   - Rakip Active’e vurup bench’e 6 damage counter dağıtma hesabı yapılır.
   - Fezandipiti ex / Meowth ex / Budew duruma göre yardımcı saldırı/seçenek olur.

6. **Supporter / el kontrolü**
   - `Boss's Orders`: hedef Pokémon’u öne çekmek için.
   - `Lillie's Determination`: eli yenilemek.
   - `Judge`: rakip elini bozmak / kendi elini dengelemek.
   - `Unfair Stamp`: KO sonrası rakip elini kısmak.
   - `Team Rocket's Watchtower`: stadium kontrolü.

7. **Search / AI’sız fark**
   - `bots/dragapult_lean4`: `SEARCH_ENABLED = True`
     - Rule policy’nin top adayları üzerinde 1-ply lookahead.
     - Hidden zone determinizasyonu + greedy rollout.
   - `bots/dragapult_lean4_noai`: `SEARCH_ENABLED = False`
     - Sadece yukarıdaki kural skorları.
     - ML/neural/value/policy-prior yok.

## Kısa özet

Lean4 pilotu:

**Dreepy/Drakloak/Dragapult setup → doğru enerji → Phantom Dive + bench damage → Boss/Judge/Lillie ile tempo → Crushing Hammer ile enerji bozma**

üzerine kurulu, tamamen kural tabanlı bir Dragapult pilotudur.

Top2 deck’i için ayrı bir **Munkidori/Chi-Yu/Crispin/Judge/Unfair Stamp** policy’si gerekir; lean4 kodu o deck’e doğrudan taşınmaz.
