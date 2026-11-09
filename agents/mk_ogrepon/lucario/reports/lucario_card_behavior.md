# Mega Lucario — kart davranışı doğrulaması

Kaynak: `pokemon/ptcg_engine_playground/ptcgProgram 22/CardImpl.h`,
`cg.api` kart verisi ve `replays/` içindeki Mega Lucario (678) içeren 115
episode. Bu rapor **hafızadan yazılan kart metinlerinin** motorla
karşılaştırılmasıdır.

## Kritik bulgu: Poffin + Riolu 677 uyumsuz

`Buddy-Buddy Poffin` (1086) motor kodu:

```text
.effectDeckToBenchAndShuffle(2).targetBasicPokemon()
.targetCondition(TargetType::MaxHp, 70, LessEqual)
```

Yani yalnızca **Basic ve HP ≤ 70** Pokémon arar.

Kullanıcı destesindeki hedefler:

| Kart | ID | HP | Poffin alır mı? |
|---|---:|---:|---|
| Riolu | 677 | 80 | **Hayır** |
| Makuhita | 673 | 80 | **Hayır** |
| Lunatone | 675 | 110 | Hayır |
| Solrock | 676 | 110 | Hayır |
| Budew | 235 | 30 | Evet |

Yani `4× Riolu 677 + 4× Poffin` kombinasyonunda Poffin pratikte sadece
**tek Budew**'i arayabilir; diğer 3 Poffin ölü karttır.

Gerçek Mega Lucario replay'lerinde Poffin, **HP 70 Riolu 333/974** kartlarını
arıyor:

```text
episode-125660340: Poffin -> Riolu 333, Riolu 333
episode-125907518: Poffin -> Riolu 974, Riolu 333
episode-125311599: Poffin -> Riolu 333, Riolu 333
```

### İki tutarlı seçenek

1. **Poffin kalsın:** 4× Riolu 677 yerine 4× Riolu **333 veya 974** (HP 70).
2. **Riolu 677 kalsın:** 4× Poffin yerine 4× **Poké Pad (1152)** veya
   **Dusk Ball (1102)** kullan.

Gerçek Petit Canard listeleri de bu ayrımı doğruluyor:
- `episode-125644927`: Riolu 677 + Poké Pad + Dusk Ball, Poffin yok.
- `episode-125500450`: Poffin + Riolu 974/333.

## Mega Lucario ex (678)

| Özellik | Motor değeri |
|---|---|
| Stage | Stage1, MegaEx |
| HP | 340 |
| Weakness | **Psychic (5)** |
| Retreat | 2 |
| Prize | **3** |
| Evrim | Riolu'dan |
| Aura Jab (982) | 130, {F}; discard'ten **3 Basic {F} Energy**'ye kadar bench'e tak |
| Mega Brave (983) | 270, {F}{F}; sonraki tur bu saldırı kullanılamaz |

- **Weakness Psychic doğru.** Alakazam ve Psychic saldırıları / Munkidori
  Mega Lucario'ya karşı tehlikeli.
- Aura Jab'ın enerji takması 115 episode içinde **625 Aura Jab** kullanımında
  gözlendi; hemen sonraki loglarda bench'e **Basic {F} (cardId 6)** attach
  olayları görüldü (30 doğrudan gözlem).
- **Mega Brave 243** kez kullanıldı.
- Mega ex olduğu için 3 prize verir.

## Diğer kartlar

| Kart | Motor davranışı | Not / gözlem |
|---|---|---|
| Riolu 677 | Basic, 80 HP, 1 enerji ile 30; sonraki tur aynı saldırı yok | Poffin hedefi değil |
| Riolu 333/974 | Basic, **70 HP**, Poffin hedefi | Gerçek Poffin listelerinde kullanılıyor |
| Makuhita 673 | Basic, 80 HP; 10 / 30 hasar | Poffin hedefi değil |
| Hariyama 674 | Stage1, 150 HP; evolve ability **Heave-Ho Catcher** (rakip bench'ten gust); Wild Press 210, kendine 70 | Tek prize; Mega'ya alternatif |
| Lunatone 675 | Basic, 110 HP; **Lunar Cycle** (Solrock sahadaysa Basic F discard, 3 çek); Power Gem 50 | 115 episode'da **771** Lunatone ability kullanımı |
| Solrock 676 | Basic, 110 HP; Cosmic Beam 70, bench'te Lunatone yoksa saldırı başarısız; weakness/resistance yok | Lunatone ile birlikte |
| Budew 235 | Basic, 30 HP; Itchy Pollen 10 + rakip item kilidi | Poffin'in tek hedefi (677 destesinde) |
| Premium Power Pro 1141 | Bu tur F Pokémon saldırıları aktife +30 | Saldırı öncesi oynanmalı |
| Fighting Gong 1142 | Basic F Pokémon veya Basic F Energy arar | Riolu/Makuhita/Lunatone/Solrock/energy |
| Hero's Cape 1159 | +100 max HP; **ACE SPEC** | 1 adet; Mega/Hariyama'ya |
| Switch 1123 | Kendi aktifini bench ile değiştirir | |
| Boss's Orders 1182 | Rakip bench'ten gust | KO/setup hedefi |
| Lillie's Determination 1227 | 6 çek; tam 6 prize ise 8 | |
| Ultra Ball 1121 | 2 discard, Pokémon ara | |
| Night Stretcher 1097 | Pokémon veya **Basic** enerji al | Rock F special alınamaz |
| Basic {F} Energy 6 | Basic Fighting | 12 adet |
| Rock Fighting Energy 20 | Special, F sağlar, F Pokémon'a attack effect immunity | **Kullanıcı destesinde yok**; doğru |

## Gerçek replay action dağılımı (115 episode, 678 kullanan taraf)

| Aksiyon | Kullanım |
|---|---:|
| Lunatone ability | 771 |
| Aura Jab | 625 |
| Mega Brave | 243 |
| Hariyama evolve | 0 (label yakalanamadı; kaynakta var) |
| Cosmic Beam | gözlemlendi |
| Wild Press | gözlemlendi |

## Sonuç

- Kullanıcı destesindeki **Poffin / Riolu 677** uyumsuzluğu gerçek bir
  tutarsızlık; düzeltilmeden 4 kart ölü kalıyor.
- Mega Lucario'nun Psychic weakness'ı doğru; Alakazam eşleşmesi gerçekten zor.
- Basic F only kararı doğru; Rock Fighting çıkarılmış.
- Aura Jab / Mega Brave / Lunatone / Solrock / Hariyama motor davranışı
  kaynakla uyumlu.
