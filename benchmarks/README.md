# Benchmarks — bizim olmayan deckler ve referans politikalar

Bu klasör, kendi botlarımızı karşılaştırmak için kullanılan **dış kaynaklı
deckleri ve referans politikaları** içerir.

- **Bizim seçtiğimiz botlar:** `bots/` altında (`dragapult`, `lucario`, `banette`).
- **Buradaki decklerin çoğu:** Kaggle replay’lerinden veya Limitless/online
  turnuva listelerinden çıkarıldı.

## Klasörler

```
benchmarks/
  external_agents/   # güçlü public referans ajanlar (politikalarıyla)
  meta_decks/        # replay/online deckleri + basit lab politikaları
```

### `external_agents/`

Bunlar **bizim kodumuz değil**; `wmh/ptcg-abc` public reposundan alınmış
referans politikalardır:

- `alakazam`
- `garchomp`
- `bellibolt`
- `grimmsnarl`
- `dragapult_sample` (resmî sample politika)
- `lucario_mega_hariyama`

Amaç: yeni politikanı bunlara karşı ölçmek. Bu ajanlar güçlü politikalardır;
template bot değildir. Kaynak/atıf: [`external_agents/README.md`](external_agents/README.md).

### `meta_decks/`

Kaggle replay’lerinden / online listelerden çıkarılan deckler:

- `ns_zoroark`
- `basic_box`
- `hydrapple`
- `festival_lead`
- `ceruledge`
- `mega_excadrill`
- `mega_abomasnow`
- `slowking_kangaskhan` (sadece deck; politika yok)
- `mega_kangaskhan` (sadece deck; politika yok)

Buradaki `main.py` dosyaları **basit lab politikaları**; deckin gerçek gücünü
temsil etmez. Deck karşılaştırması için değil, “sabit ve tekrarlanabilir rakip”
olarak kullanılmalı. Kaynak/uyarı: [`meta_decks/README.md`](meta_decks/README.md).

## Nasıl çalıştırılır?

```bash
# tek eşleşme
python3 tools/arena.py \
  --bot0 bots/dragapult \
  --bot1 benchmarks/external_agents/alakazam \
  --games 100

# external agent turnuvası
python3 tools/tournament.py \
  --bots-dir benchmarks/external_agents \
  --games 20

# meta deck lab turnuvası
python3 tools/tournament.py \
  --bots-dir benchmarks/meta_decks \
  --games 20
```

`slowking_kangaskhan` ve `mega_kangaskhan` klasörlerinde `main.py` yok; turnuva
onları otomatik atlar (deck-only referans).

## Kısa notlar

- **Dış ajanlar güçlü**, lab politikaları zayıf. Bir deckin kaybediyor olması
  deckin kötü olduğu anlamına gelmez; politika farkı olabilir.
- `external_agents` kodunu public repoya taşımadan önce upstream lisansını
  kontrol et; bu klasör özel takım içi test için.
- Deck listeleri Pokémon Elements; competition-use only, public dağıtımda dikkat.
- Bizim botlar: `bots/dragapult` (AI çalışması için ana), `bots/lucario`,
  `bots/banette`.
