# Meta deck listeleri

Bu klasördeki listeler, 20 Ekim 2026'da Kaggle'daki halka açık replay'lerden
çıkarıldı; 24 Ekim 2026'da 82 kendi maçımız + 60 üst-sıra takım replay'i ile
güncellendi.

## Kaynak

- **188 episode / 376 deck**
- Leaderboard'daki ilk 15 takımın en iyi 2 submission'ından, her birinden
  ilk 8 episode (tekrarlar ayıklandı)
- Replay'lerden deck'ler `steps[0][0].visualize[0].action` alanından okundu
- Aynı takımın birden çok maçı olduğu için bu bir **örneklem**; tam ladder
  census'u değil

## Dağılım

| Archetype | Deck | Pay | W-L | Win rate |
|---|---:|---:|---:|---:|
| Alakazam | 97 | 25.8% | 42-55 | 43.3% |
| Festival Lead | 64 | 17.0% | 36-28 | 56.2% |
| Dragapult | 63 | 16.8% | 35-28 | 55.6% |
| Banette | 57 | 15.2% | 36-21 | 63.2% |
| Hydrapple | 28 | 7.4% | 10-18 | 35.7% |
| Mega Kangaskhan | 27 | 7.2% | 14-13 | 51.9% |
| Slowking-Kangaskhan | 14 | 3.7% | 7-7 | 50.0% |
| Ceruledge | 8 | 2.1% | 5-3 | 62.5% |
| Diğer | 18 | 4.8% | - | - |

Öne çıkan matchup döngüsü:

- Alakazam 5-0 Slowking
- Slowking 2-1 Banette
- Banette 2-1 Dragapult, 2-0 Alakazam
- Dragapult 3-1 Alakazam

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `banette.csv` | 57/57 Banette deckinin birebir aynı consensus listesi |
| `banette_kieran_bloodmoon.csv` | İlk deneme: 2 Kieran + 2 Bloodmoon, Patrat korundu (Legacy Energy) |
| `banette_kieran_bloodmoon_neutral.csv` | Final bot: aynı liste + Neutralization Zone (Legacy yerine) |
| `alakazam.csv` | En sık görülen Alakazam listesi |
| `dragapult.csv` | En sık görülen Dragapult listesi |
| `festival_lead.csv` | Grookey/Thwackey/Dipplin + Festival Grounds |
| `mega_kangaskhan.csv` | Mega Kangaskhan listesi |
| `slowking_kangaskhan.csv` | Slowking + Mega Kangaskhan listesi |
| `ceruledge.csv` | Ceruledge ex listesi |
| `dragapult_blaziken.csv` | Blaziken ex / Chi-Yu / Shaymin Dragapult varyantı |
| `dragapult_top4.csv` | 4 Fire / 4 Psychic / 2 Dark, 4 Poke Pad Dragapult varyantı |
| `mega_excadrill_top_a.csv` | Genesect / Metagross / Mega Excadrill top varyantı |
| `mega_excadrill_top_b.csv` | Team Rocket's Transceiver / Gravity Mountain varyantı |
| `mega_starmie.csv` | Staryu + Mega Starmie ex / Mega Froslass ex |
| `mega_froslass.csv` | Snorunt + Mega Froslass ex |
| `teal_mask_ogerpon.csv` | Teal Mask Ogerpon ex / Tera Orb |
| `munkidori_crustle.csv` | Munkidori + Crustle + Froslass |
| `crustle_fire.csv` | Dwebble/Crustle + Centiskorch |
| `ns_zorua.csv` | N's Zorua / N's Zoroark ex / Xerosic / N's Castle |
| `archaludon.csv` | Replay’den çıkarılan Archaludon listesi |

## İlk bot: Banette + Kieran + Bloodmoon + Neutralization Zone

Consensus Banette listesine göre:

- `-1 Prism Tower`
- `-1 Pokégear 3.0`
- `-1 Legacy Energy`
- `+1 Kieran` (toplam 2)
- `+1 Bloodmoon Ursaluna ex` (toplam 2)
- `+1 Neutralization Zone`

Patrat korundu. Liste 60 kart, 1 ACE SPEC (Neutralization Zone), engine
tarafından kabul ediliyor. Amaç: Kangaskhan/Ceruledge'e karşı 2. Bloodmoon +
Kieran ile daha iyi 2-vuruş planı; düşük ladder'da çok kalabalık olan
Mega Abomasnow'un ex saldırılarını Neutralization Zone ile sıfırlamak.

Lokal testlerde (BanettePolicy):

- Mega Abomasnow: %39 → **%57**
- Mega Kangaskhan: %55 → %57
- Alakazam/Dragapult/Festival: hafif iyileşme
- Slowking: %56 → %44 (Neutralization'ın bedeli)

## 24 Ekim 2026 güncellemesi

- 82 kendi maçımız: Dragapult 101 deck, Alakazam 13, Mega Abomasnow 10,
  Mega Kangaskhan 9, Mega Lucario 8; ayrıca Mega Starmie, Teal Mask Ogerpon,
  Munkidori/Crustle, Mega Froslass, N's Zorua gibi yeni archetype'lar.
- 60 üst-sıra takım replay'i: Dragapult 26, Mega Kangaskhan 23, Hydrapple 18,
  Mega Excadrill 16, Banette 16, Alakazam 11, Slowking-Kangaskhan 8.
- Üst-sıra Dragapult pilotları 26/26 **first** seçti.
- Yeni listeler `tools/build_belief_data.py` ile botun self-contained belief
  kütüphanesine aktarılır.

## Doğrulama

```bash
python3 tools/check_deck.py decks/meta/banette_kieran_bloodmoon.csv --engine
python3 tools/check_deck.py bots/banette/deck.csv --engine
python3 tools/check_deck.py decks/meta/ns_zorua.csv --engine
python3 tools/build_belief_data.py
```

Replay analizi:

```bash
python3 tools/replay_meta.py summarize replays/*.json
python3 tools/replay_meta.py decks replays/*.json --archetype Banette
```
