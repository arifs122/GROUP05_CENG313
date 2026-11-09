# Mega Lucario vs Dragapult — kazanan/kaybeden erken plan

Kaynak: `replays/**/*.json`, Mega Lucario (678) vs Dragapult (121/120) içeren
**91 episode**: **52 kazanan**, **39 kaybeden**. İlk 4 tur MAIN aksiyonları.

## Kazanan pilotun ilk 3 turu

- İlk aktif: **Riolu 25**, **Solrock 19**, Lunatone 5, Makuhita 3.
- İlk bench: **Lunatone 19**, Riolu 9, Solrock 9, Makuhita 7.
- İlk 3 tur aksiyon: PLAY 827, ATTACH 201, ABILITY 117, ATTACK 111,
  EVOLVE 71, END 82.
- Enerji hedefleri: **Riolu 110**, **Solrock 67**, Mega 52, Makuhita 10,
  Lunatone 8, Hariyama 5.
- En çok oynananlar: Fighting Gong 127, Riolu 117, **Dusk Ball 92**,
  Lunatone 87, **Poké Pad 86**, Makuhita 86, Lillie 84, **Solrock 71**,
  Carmine 49, Premium Power Pro 42.
- Saldırılar: **Cosmic Beam 66**, **Aura Jab 57**, Accelerating Stab 32,
  Mega Brave 5.
- Ability: **Lunatone 157**.

## Kaybeden pilotun ilk 3 turu

- İlk aktif: Riolu 12, Solrock 11, Lunatone 7, Makuhita 7.
- İlk bench: Lunatone 12, Riolu 11, Makuhita 5, Solrock 3.
- İlk 3 tur aksiyon: PLAY 548, ATTACH 150, ABILITY 66, ATTACK 63, EVOLVE 46.
- Enerji: Riolu 73, Mega 39, Solrock 36, Makuhita 23.
- Oyunlar: Riolu 103, Fighting Gong 85, Poké Pad 69, Lillie 63,
  Dusk Ball 56, Makuhita 50, Solrock 47, Lunatone 44.
- Saldırılar: Cosmic Beam 32, Aura Jab 24, Accelerating Stab 10,
  Quick Attack 7, Mega Brave 4.
- Ability: Lunatone 89.

## Çıkarılan plan kuralları

1. **Solrock + Lunatone motorunu erken kur.** Kazananlar Solrock'u hem aktif
   hem bench olarak kullanıyor; Solrock/Lunatone basic skoru artırıldı (680).
2. **Lunatone ability her tur.** Kazananlarda 157, kaybedenlerde 89;
   ability 980 puana çıkarıldı, saldırıdan önce.
3. **Enerjiyi sadece Mega'ya değil Riolu/Solrock'a da bağla.** Kazananlarda
   Solrock 67 attach; attach hedef skoru Solrock/Lunatone için 470.
4. **Aura Jab'ı erken kullan.** Kazananlarda Aura Jab 57; Mega Brave 5.
   Aura Jab'a erken +120, Mega Brave'a erken -80.
5. **Arama kartlarını erken oyna.** Dusk Ball/Poké Pad/Fighting Gong
   kazananlarda daha yüksek; arama skoru 720-730, attach'tan önce.
6. **Cosmic Beam'i kullan.** Solrock/Lunatone bench'i varken 70 hasar,
   weakness/resistance yok; kaybedenlerde daha az.
7. **Carmine/Lillie ile kart akışı.** Carmine 760, Lillie 760.

Bu kurallar sonrası bizim pilot + `deck_real_petit` Dragapult proxy'ye karşı
200 maçta **%26.5-41.5** bandına çıktı (seed/RNG'ye göre); önceki %15-23'tü.
Hedef %30 alt sınırı bazı koşularda geçildi, ancak mirror hâlâ %40'ın altında.
