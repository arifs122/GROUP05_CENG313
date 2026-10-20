# Meta deck listeleri

Bu klasördeki listeler, 20 Ekim 2026'da Kaggle'daki halka açık replay'lerden
çıkarıldı.

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
| `banette_kieran_bloodmoon.csv` | İlk botumuzun listesi: 2 Kieran + 2 Bloodmoon, Patrat korundu |
| `alakazam.csv` | En sık görülen Alakazam listesi |
| `dragapult.csv` | En sık görülen Dragapult listesi |
| `festival_lead.csv` | Grookey/Thwackey/Dipplin + Festival Grounds |
| `mega_kangaskhan.csv` | Mega Kangaskhan listesi |
| `slowking_kangaskhan.csv` | Slowking + Mega Kangaskhan listesi |
| `ceruledge.csv` | Ceruledge ex listesi |

## İlk bot: Banette + Kieran + Bloodmoon

Consensus Banette listesine göre:

- `-1 Prism Tower`
- `-1 Pokégear 3.0`
- `+1 Kieran` (toplam 2)
- `+1 Bloodmoon Ursaluna ex` (toplam 2)

Patrat korundu. Liste 60 kart, 1 ACE SPEC (Legacy Energy), engine tarafından
kabul ediliyor. Amaç: Kangaskhan/Ceruledge'e karşı 2. Bloodmoon + Kieran ile
daha iyi 2-vuruş planı; Slowking'e karşı Patrat ve daha iyi Boss hedefi.

Lokal testlerde (BanettePolicy, 100 maç ilk/second ayrı) bu varyant:

- İyi matchup'larda (Alakazam/Festival/Dragapult) base'e yakın
- Kötü matchup'larda (Kangaskhan/Slowking/Ceruledge) base'den daha iyi
- İlk/second arasında dengeli

## Doğrulama

```bash
python3 tools/check_deck.py decks/meta/banette_kieran_bloodmoon.csv --engine
python3 tools/check_deck.py bots/banette/deck.csv --engine
```

Replay analizi:

```bash
python3 tools/replay_meta.py summarize replays/*.json
python3 tools/replay_meta.py decks replays/*.json --archetype Banette
```
