# Mega Lucario destesi

`agents/mk_ogrepon/lucario/deck.csv` — kullanıcı listesi, 60 kart.

```bash
python3 tools/check_deck.py agents/mk_ogrepon/lucario/deck.csv --engine
# cards: 60  unique: 17
# basic Pokémon: 14
# ACE SPEC: [(1159, "Hero’s Cape", 1)]
# engine: deck accepted
# OK: deck looks legal
```

- Pokémon 20: Riolu 677 ×4, Mega Lucario ex 678 ×3, Makuhita 673 ×4,
  Hariyama 674 ×3, Lunatone 675 ×3, Solrock 676 ×2, Budew 235 ×1.
- Trainer 28: Poffin ×4, Ultra Ball ×4, Night Stretcher ×2, Switch ×2,
  Boss ×3, Lillie ×4, Premium Power Pro ×4, Fighting Gong ×4, Hero’s Cape ×1.
- Enerji 12: Basic {F} ×12.
- 1 ACE SPEC, Radiant yok.

## Uyumsuzluk

`Buddy-Buddy Poffin` yalnız Basic HP ≤70 arar. Bu destede Riolu 677 (80) ve
Makuhita (80) hedef değil; Poffin yalnız Budew (30) alabilir. Ayrıntı:
[`lucario_card_behavior.md`](lucario_card_behavior.md).

## Varyantlar

| Dosya | Fark |
|---|---|
| `deck_riolu974.csv` | Riolu 677 → 974 (HP70); Poffin çalışır |
| `deck_pokepad.csv` | Poffin → Poké Pad (1152); Riolu 677 kalır |
| `deck_974_pokepad.csv` | İkisi birden |
| `deck_real_petit.csv` | Replay'den Petit Canard Mega Lucario listesi |

Hepsi `check_deck.py --engine` ile legal.
