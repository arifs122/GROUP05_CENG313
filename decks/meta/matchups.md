# Matchup notları — neden kazanıyorlar, nasıl çözülür?

Bu dosya 188 public replay, güçlü politikalı rakip ajanlar ve kart havuzu
taramasından çıkan notları bir araya getirir.

## 1. Yaygın deckler neden kullanılıyor?

| Deck | Ana motor | Üstünlük | Zayıflık | İlk çözüm |
|---|---|---|---|---|
| **Mega Abomasnow** | 35 Water, Waitress/Mega Signal, Frost Barrier | 350 HP; 200 hasar + -30; sample deck herkeste var; 1-prize saldırganları tek vuruşta alır | Metal weakness; 3-prize; yavaş | Metal saldırgan (Mega Excadrill/Steven's Metagross), tek vuruş (Mega Emboar/Meganium), Crustle, Neutralization Zone |
| **Mega Lucario** | Aura Jab ile discard'dan 3 Fighting; Solrock/Lunatone draw; Hariyama gust | 340 HP; 270 hasar; enerji hızlandırma; gust | Psychic weakness; 3-prize; Mega Brave cooldown | Psychic saldırgan (Mega Gardevoir), Alakazam counter, Crustle/Neutralization |
| **Dragapult** | Drakloak motoru; Crispin; Phantom Dive | 320 HP; 200 + 6 bench counter; çoklu KO; weakness yok | Banette Hide 'n' Sneak; Fairy Zone + Psychic; Crustle; bench koruması | Banette, Lillie's Clefairy + Psychic, Crustle, Battle Cage/Shaymin |
| **Alakazam** | Dudunsparce draw; 20×hand Powerful Hand | Single-prize; 300-400 hasar; weakness/defense bypass (counter) | 140 HP; effect immunity; item lock; hand disruption | Banette, Mist/Rock Energy, Battle Cage, Judge/Eri, 140 HP'yi tek vuruş |
| **Cynthia's Garchomp** | Gabite Champion's Call; 0 retreat; 260 | Tutarlılık; 330 HP; 260 hasar; Corkscrew ile el doldurma | Grass weakness; 2-prize; enerji discard | Grass (Mega Meganium/Hydrapple), tek vuruş, Boss Gabite |
| **Mega Kangaskhan** | Run Errand draw; 300 HP; 200+ | Colorless; draw; yüksek HP | Fighting weakness; coin flip; 3-prize; Alakazam/Slowking | Lucario/Zygarde, Flutter Mane/Iron Thorns, Crustle/Neutralization |
| **Festival Lead** | Festival Grounds + Dipplin/Thwackey; çift saldırı | Single-prize; aynı turda 2 KO; Thwackey search | Fire weakness; bench spread; Banette | Fire (Mega Emboar/Charizard), spread, Banette |
| **Banette** | Hide 'n' Sneak; Gwynn; Dhelmise/Sinistcha discard scaling | Effect immunity; Alakazam/Dragapult'a iyi; single-prize | Abomasnow/Kangaskhan damage; yavaş; Darkness weakness | Abomasnow/Kangaskhan, Dark saldırgan, Boss |
| **Bellibolt** | Electric Streamer; 230 | Lightning energy acceleration; Kilowattrel non-ex | Fighting weakness; Garchomp/Lucario | Lucario/Garchomp, ability lock |
| **Grimmsnarl** | Punk Up; Munkidori/Froslass chip | 180+30 bench; Dark acceleration | Grass weakness; Garchomp | Grass, tek vuruş |

## 2. Counter araçları

### Tip avantajı
- Abomasnow (Water) → **Metal**: Mega Excadrill, Steven's Metagross, Mega Mawile.
- Lucario (Fighting) → **Psychic**: Mega Gardevoir, Mega Floette, Mega Diancie.
- Alakazam (Psychic) → **Darkness**: Mega Gengar, Grimmsnarl, Mega Absol.
- Garchomp (Fighting/Dragon) → **Grass**: Mega Meganium, Hydrapple.
- Kangaskhan (Colorless) → **Fighting**: Lucario, Mega Zygarde.
- Festival (Grass) → **Fire**: Mega Emboar, Mega Charizard.
- Grimmsnarl (Dark) → **Grass**.
- Bellibolt (Lightning) → **Fighting**.

### Effect / hasar counter'ları
- **Hide 'n' Sneak** (Banette/Shuppet/Poltchageist/Sinistcha): attack/ability effect'lerini engeller; Alakazam counter'ları ve Dragapult bench counter'ları.
- **Crustle**: ex saldırılarına tam bağışıklık; Abomasnow/Lucario/Dragapult/Kangaskhan'ı duvar.
- **Neutralization Zone**: non-Rule-Box Pokémon'a ex hasarını sıfırlar; Abomasnow/Kangaskhan/Lucario/Dragapult.
- **Battle Cage / Shaymin**: bench damage counter'larını engeller; Dragapult/Alakazam.
- **Mist Energy / Rock Fighting Energy**: attack effect'lerini engeller; Alakazam/Banette.
- **Flutter Mane / Iron Thorns**: ability lock; Kangaskhan Run Errand, Fezandipiti, Grimmsnarl.
- **Jellicent ex / Tyranitar / Team Rocket's Arbok**: item/ability lock.
- **Patrat**: damage counter'ların taşınmasını engeller; Munkidori/Orbeetle.

### Tek vuruş / yüksek tavan
- **Mega Meganium**: 3 Grass → ~370 (Meganium Wild Growth).
- **Mega Emboar**: 320 + Maximum Belt 50 → 370; 380 HP.
- **Mega Gardevoir**: 50 × Psychic enerji; 8 enerji → 400.
- **Mega Dragonite**: 330 + Belt 50 → 380; 370 HP.
- **Mega Excadrill**: 200 × 2 (Abomasnow Metal weak) → 400.
- **Mega Lucario**: 270 + Premium/Belt/Kieran ile 300+.

## 3. Üç deck planı

### A. Dragapult — generalist / ilk competitive bot
- Deck: `decks/meta/dragapult.csv` (24 unique).
- Motor: Dreepy → Drakloak → Dragapult; Drakloak Recon Directive; Crispin enerji; Munkidori/Fezandipiti destek; Crushing Hammer.
- Plan: Phantom Dive ile 200 + 6 bench counter; birden fazla KO; 2-prize ama 320 HP.
- İyi: Alakazam, Abomasnow, Festival, Banette (güçlü pilotla).
- Kötü: Kangaskhan, bazı control deckleri.
- Neden ilk: güçlü pool'da %73.6, hiçbir matchup %55 altı değil.

### B. Lucario — anti-Abomasnow
- Deck: Lucario listesi (4 Mega Lucario, 3 Riolu, 3 Solrock/2 Lunatone, 2 Hariyama, 13 Fighting).
- Motor: Aura Jab discard'dan 3 Fighting; Solrock/Lunatune draw; Hariyama gust; Premium Power Pro.
- Plan: Aura Jab ile enerji kur, Mega Brave 270; Hariyama ile bench hedefi.
- İyi: Abomasnow %82, Banette %90, Bellibolt/Grimmsnarl.
- Kötü: Alakazam %35, Dragapult %40.

### C. Mega Excadrill — Metal anti-tank / Mega varyantı
- Deck: 4 Beldum/4 Metang/3 Drilbur/2 Mega Excadrill; 2 Genesect ex; Metagross; 16 Metal; Iron Defender/Jumbo Ice Cream/Philippe.
- Motor: Metang/Genesect ile Metal enerji; Drilbur → Mega Excadrill; Maximum Drilling 200 (+extra enerji).
- Abomasnow: 200 × 2 (Metal weak) = 400 → tek vuruş. 340 HP.
- Zayıf: Fire (Mega Emboar/Charizard/Chandelure), hızlı Psychic/effect deckleri.
- Politika yazılacak: Metang/Genesect ability, Mega Excadrill enerjisi, Boss engine hedefi, Iron Defender/Jumbo.

## 4. Mega varyantları — lab notları

- **Mega Meganium + Meganium**: combo gerçek (3 Grass → 370). İlk testte Abomasnow %52, ama Alakazam %8, Lucario %4, Dragapult %26. Stage 2 kurulumu ve 70 HP Chikorita sorunu; Ogerpon aktif + daha iyi arama ile tekrar denenebilir.
- **Mega Emboar**: 380 HP, 320 + Belt 370; Oricorio ex/Blaziken ex/Ethan's Ho-Oh/Firebreather ile hızlandırma. Abomasnow'u tek vuruş; Fire weakness Water. Henüz deck/politika yazılmadı.
- **Mega Gardevoir**: Overflowing Wishes + Mega Symphonia; Mega Floette/Aromatisse/Wondrous Patch/Telepath ile Psychic enerji. 8 enerji → 400. Stage 2; kurulum riski.
- **Mega Dragonite**: 330 + Belt 380; Dragonair Evolutionary Guidance; Crispin/Energy Switch. 370 HP.
- **Crustle**: ex immunity; Abomasnow/Lucario/Dragapult/Kangaskhan duvarı; Alakazam/Festival'e zayıf.
- **Banette**: Alakazam/Dragapult counter'ı; Abomasnow/Kangaskhan'a zayıf. Lab control olarak tutulacak.

## 5. Test protokolü

- Rakip: güçlü politikalar (Alakazam, Dragapult official sample, Lucario, Garchomp, Bellibolt, Grimmsnarl) + custom Abomasnow.
- En az 100 maç; BO3; ilk/second ayrı.
- Template botla alınan sonuçlar sadece crash/legal kontrolü; matchup kanıtı değil.
- Son hakem Kaggle ladder; local sadece eleme.
