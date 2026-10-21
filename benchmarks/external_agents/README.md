# External reference agents

Buradaki ajanlar **bizim kodumuz değil**. `wmh/ptcg-abc` public reposundan
alınmış referans politikalardır; yeni politikanı ölçmek için rakip olarak
kullanılır.

Kaynak: <https://github.com/wmh/ptcg-abc>
Alınma tarihi: 21 Ekim 2026.

## İçerik

| Klasör | Deck | Not |
|---|---|---|
| `dragapult_sample` | Dragapult | Resmî Kaggle sample pilotunun sarılmış hâli |
| `alakazam` | Alakazam + Dudunsparce | Güçlü, single-prize |
| `lucario_mega_hariyama` | Mega Lucario + Hariyama | Güçlü, Abomasnow avcısı |
| `garchomp` | Cynthia's Garchomp | `policy_base.py` gerekir |
| `bellibolt` | Iono's Bellibolt | `value_weights.py` gerekir |
| `grimmsnarl` | Marnie's Grimmsnarl | `generic_policy.py` + `policy_base.py` |

`lucario_mega_hariyama` ve `dragapult_sample`, `bots/` altındaki kendi
seçtiğimiz botlarla aynı kaynak kodu paylaşır; burada **karşılaştırma için
kopya** olarak duruyor.

## Kullanım

```bash
python3 tools/arena.py \
  --bot0 bots/dragapult \
  --bot1 benchmarks/external_agents/alakazam \
  --games 100

python3 tools/tournament.py \
  --bots-dir benchmarks/external_agents \
  --games 20
```

## Uyarı

- Bu kod bize ait değil; upstream repo’nun lisans/şartları geçerlidir.
- Repo’yu public yapacaksan bu klasörü çıkarmayı veya upstream lisansını
  kontrol etmeyi unutma.
- Deck listeleri Pokémon Elements kapsamındadır; competition-use only.
