# Deste — Top1 replay çıkarımı

Komut: `python3 agents/mk_ogrepon/tools/extract_deck.py`

- Kaynak: `replays/top5_nov1/17014244/*.json` (takım:
  `やる気元気ミワハルキ 2nd ver.`), sadece takımın kazandığı replay'ler.
- 50 galibiyet replay'i tarandı.
- **50/50 aynı 60 kartlık liste** çıktı; farklı varyant yok.
- Step 0 submission action'ı `steps[0][0].visualize[0].action` üzerinden
  `replay_meta.extract_decks()` ile okundu.
- Örnek replay: `episode-125781222-replay.json`.

## Liste

```
11x Basic {G} Energy (1)
 4x Prism Energy (16)
 1x Chi-Yu (31)
 4x Teal Mask Ogerpon ex (96)
 1x Wellspring Mask Ogerpon ex (108)
 1x Munkidori (112)
 1x Fezandipiti ex (140)
 2x Latias ex (184)
 1x Pecharunt (230)
 2x Lillie's Clefairy ex (272)
 2x Mega Kangaskhan ex (756)
 2x Meowth ex (1071)
 1x Unfair Stamp (1080)
 3x Bug Catching Set (1094)
 2x Night Stretcher (1097)
 4x Energy Switch (1116)
 4x Ultra Ball (1121)
 2x Tera Orb (1127)
 3x Boss's Orders (1182)
 1x Ciphermaniac's Codebreaking (1188)
 1x N's Plan (1221)
 4x Lillie's Determination (1227)
 3x Area Zero Underdepths (1250)
```

## Doğrulama

`python3 tools/check_deck.py agents/mk_ogrepon/deck_top1.csv --engine`:

```
cards: 60  unique: 23
basic Pokémon: 17
ACE SPEC: [(1080, 'Unfair Stamp', 1)]
engine: deck accepted
OK: deck looks legal
```

- 1 ACE SPEC, Radiant yok.
- Basic Pokémon kopyası 17 (10 farklı Basic satırı).
- Mega ex: 756 (Mega Kangaskhan ex); engine prize değeri 3.
- Tera: 96 ve 108; Area Zero bench limiti için gerekli.

Tam kart davranış tablosu: [`card_behavior.md`](card_behavior.md).
