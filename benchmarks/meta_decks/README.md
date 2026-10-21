# Meta decks — replay/online kaynaklı deckler

Buradaki deck listeleri bizim icat ettiğimiz deckler değil; Kaggle
replay’lerinden ve online turnuva listelerinden çıkarıldı.

- **188 public Kaggle replay**: leaderboard’daki ilk 15 takımın en iyi 2
  submission’ından, ilk 8 episode (tekrarlar ayıklandı).
- **Online turnuva listeleri**: Limitless / TCG Contender / pokemoncard.io
  (N’s Zoroark ve Basic Box için).

## İçerik

| Deck | Kaynak | Politikası |
|---|---|---|
| `ns_zoroark` | Limitless N’s Zoroark listesi | Basit lab politikası |
| `basic_box` | Limitless Basic Box listesi | Basit lab politikası |
| `hydrapple` | Kaggle replay’lerinden exact liste | Basit lab politikası |
| `festival_lead` | Kaggle replay’lerinden exact liste | Basit lab politikası |
| `ceruledge` | Kaggle replay’lerinden exact liste | Basit lab politikası |
| `mega_excadrill` | Kaggle replay’lerinden exact liste | Basit lab politikası |
| `mega_abomasnow` | Competition sample deck | Basit lab politikası |
| `slowking_kangaskhan` | Kaggle replay’lerinden exact liste | **Politika yok** |
| `mega_kangaskhan` | Kaggle replay’lerinden exact liste | **Politika yok** |

> Basic Box’ta Jirachi ex kart havuzunda olmadığı için 4. Meowth ex ile
> değiştirildi.

## Politikalar hakkında

Her `main.py`, aynı `_policy.py` motoruna bir `CONFIG` verir:

```python
from _policy import make_agent
CONFIG = {...}
agent = make_agent(CONFIG)
```

Bu politikalar **bilinçli olarak basit**: setup/evolve/play/attack skorları,
enerji hedefleri ve arama öncelikleri var. Ama top-tier pilot değiller.

- Bir deckin burada kötü görünmesi deckin kötü olduğu anlamına gelmez.
- Deck karşılaştırması yapacaksan aynı politikayı kullan veya decke özel
  politika yaz.
- `slowking_kangaskhan` ve `mega_kangaskhan` deck-only; turnuva onları atlar.

## Kullanım

```bash
python3 tools/arena.py \
  --bot0 bots/dragapult \
  --bot1 benchmarks/meta_decks/mega_abomasnow \
  --games 50

python3 tools/tournament.py \
  --bots-dir benchmarks/meta_decks \
  --games 20
```

## Kaynak/atıf

- Kaggle replay’leri: competition sayfası > Leaderboard / Submissions.
- Limitless: <https://limitlesstcg.com/decks/lists>
- TCG Contender meta: <https://tcgcontender.com/pokemon/meta>
- Deck listeleri Pokémon Elements; competition-use only.
