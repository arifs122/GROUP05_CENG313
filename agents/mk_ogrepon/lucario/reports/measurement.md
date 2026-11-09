# Ölçüm altyapısı ve mirror araştırması

## 1. Engine seed gerçeği

`ApiBattleStart` (`Api.h`) seed'i `std::random_device` ile üretiyor ve
`config.deviceRand = true` bırakıyor. Public API'de seed parametresi yok.
Bu nedenle `battle_start()` çağrıları arasında **aynı el/sıra** garanti
edilemiyor. Ayrıca `deviceRand=true` iken yazı-tura ve bazı shuffle'lar
`std::random_device` kullanıyor; iki ayrı süreç aynı RNG'yi paylaşmıyor.

## 2. Fork-paired runner

`tools/paired_arena.py`:

1. Bir `battle_start` yapılır (tek gerçek deal).
2. Her aday için `os.fork()` ile süreç klonlanır; deck sırası, prize, gizli
   kartlar ve C++ RNG durumu aynen kopyalanır.
3. `GameConfig.deviceRand` alanı `BattleData` belleğinde
   `offset 563` üzerinden `false` yapılır; böylece coin/shuffle da kopyalanan
   `std::mt19937`'yi kullanır.
4. `lucario_agent.choose()` deterministic tie-break kullanır.

Doğrulama: aynı policy iki aday olarak 10 deal'da **birebir aynı sonucu**
verdi (4/4). Bu, paired karşılaştırmanın çalıştığını gösterir.

Önemli sınır: deck varyantları `BattleStart` anında sabitlendiği için
fork ile aynı deal üzerinde farklı deste oynanamaz. Deste karşılaştırması
**seat-swapped 200 maç/rakip** ile yapıldı; policy/ağırlık karşılaştırması
fork-paired.

## 3. Koltuk değişimi

`tools/eval_pool.py` artık maçların yarısında Lucario'yu P0, yarısında P1
oynatıyor; turn-order seçimi de ayrıca almaşık. Eski ölçümler P0-only olduğu
için terk edildi.

## 4. Mirror araştırması — sonuç

| Karşılaştırma | Sonuç |
|---|---:|
| Lucario vs kendisi (aynı kod/deste), 100 BO1 | **53-46-1** (~%53) |
| Lucario vs `benchmarks/external_agents/lucario_mega_hariyama` (onun destesi), 200 | **27-173 (%13.5)** |
| Lucario vs external Lucario, **aynı kullanıcı destesi**, 50 | **13-37 (%26)** |
| Ladder replay matrisi: Mega Lucario vs Dragapult | **62.5%** (20/32) |

Yorum:

- Self-mirror ~%50 → **koltuk/bug simetrisi bozuk değil**.
- External Lucario botu aynı desteyle %74 kazanıyor → sorun **şablon/politika
  gücü**, sadece deste değil.
- External botun destesi de daha tutarlı: 4 Mega, 4 Dusk Ball, 4 Poké Pad,
  4 Carmine, 3 Solrock; kullanıcı destesinde 4 Poffin/Riolu 677 uyumsuz.
- Ladder'da Mega Lucario Dragapult'a %62.5 yapıyor; lokal %9.5-13.5 →
  lokal politika ladder pilotunun çok altında.

"Mirror ~%10" bir bug değil; karşılaştırılan bot **aynı politika değil,
güçlü external Lucario pilotu**. Bu yüzden mirror'u ayrı **self-mirror**
ve **external mirror** olarak raporlamak gerekiyor.

## 5. 2x2 ve karar ayrışması

- 2x2: `reports/pilot_deck_2x2.md` → pilot ~20 puan, deste ~17-24 puan.
- MAIN top-1: rule %40.6 → + BC prior **%58.4**; external referans %66.9.
- Dragapult planı: `reports/dragapult_plan.md`.

## 6. Paired tuning

`tools/tune_paired.py` altı ağırlığı (`ko, attack, boss_setup, lillie,
retreat, attach`) ±%20 oynatıp train pool'da fork-paired 40 deal/rakip
yarıştırdı. En iyi aday holdout'ta:

- `attach*0.8`: train Δ +0.044 (z≈1.40), holdout Δ +0.022, z≈1.86
- z < 1.96 → **anlamlı değil; `weights.json` yazılmadı.**

Rapor: `reports/paired_tuning.json`.
