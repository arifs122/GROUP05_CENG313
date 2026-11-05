# Slowking geliştirme planı (v22 → v25)

> 5 Kasım 2026 gece. Bilgisayar kapanmadan önceki durum ve "devam" denince
> izlenecek yol. Kiralama yok, GPU yok; motor kapalı kaynak binary (CPU-only).

## 1. Durum

| Deney | Ne | Ladder |
|---|---|---|
| v21 | Banette + kayıp eşleşmelere hedefli veri | **939,0** (sıra 148) — bilinen en iyi 2. slot |
| v22 | Slowking BC, 513 episode, sıfırdan | **600,0** (sıra 382) — çekilecek |

- v22 paketi: `submissions/mazlum_v22_slowking.tar.gz` (model sha `871246e5`).
- v21 paketi: `submissions/mazlum_v21_matchups.tar.gz` (geri yükleme için hazır).
- Kota 00:00 UTC'de sıfırlanıyor; **v21 restore kullanıcı onayına bağlı**.

## 2. Teşhis — v22 neden Dragapult'a %20?

30 maçlık karar izi (`MAZLUM_TRACE`), 2.039 karar, `bots/mazlum_slowking`:

| Oynanan kart | Adet | | Saldırı | Adet |
|---|---:|---|---|---:|
| Poké Pad | 67 | | Slowking (213) | 68 |
| Slowpoke | 60 | | Kyurem Trifrost (188) | 19 |
| Lillie's Determination | 45 | | Slowking (146) | 11 |
| Ultra Ball | 43 | | Slowking (242) | 10 |
| Ciphermaniac | 41 | | Clefairy (371) | **3** |
| Academy at Night | 38 | | | |
| **Lillie's Clefairy ex** | **5** | | | |

- Enerji hedefleri: Mega Kangaskhan 84, Slowking 59, Fezandipiti 33 → ajan
  "genel beatdown" oynuyor, Slowking/Kyurem planını kurmuyor.
- **Clefairy 30 maçın 5'inde oynanıyor** — Dragapult'un tek gerçek cevabı.
- Kyurem saldırısı 19 kez kullanılıyor; hat önceliklendirilmiyor.

### Kural testi (negatif)

Clefairy'yi Dragon aktifine karşı zorla oynat (`MAZLUM_TECH_RULE=1`):
**3/40 = %7,5** · taban **6/30 = %20**. Kural varsayılan **kapalı**
(`bots/mazlum_slowking/agent.py`). Zorlama, kurulum/enerji hattını bozuyor.

**Yorum:** sorun "Clefairy'yi unutmuş" değil; ajan Dragapult'a karşı
pozisyon/prize yarışını kaybediyor. Veri + hedefli eğitim gerekli.

## 3. Veri dağılımı (513 Slowking episode)

| Rakip | Episode | İnsan galibiyeti |
|---|---:|---:|
| Dragapult | 198 | %44 |
| Mega Excadrill | 102 | %77 |
| Banette | 84 | %49 |
| Alakazam | 69 | **%23** |
| Mega Kangaskhan | 34 | %79 |
| diğer | 26 | — |

Yani Dragapult verisi bol (198 episode) ama BC ajanı yerel Dragapult botuna
%20'de kalıyor → darboğaz veri miktarı değil, **BC kalitesi + yerel bot
seviyesi**.

## 4. Fazlar

### S1 — Hard-matchup fine-tune (v24, bitti)

- Veri: Slowking vs **Dragapult + Alakazam**, 267 episode → 90.953 satır.
- v22 ep23'ten fine-tune, lr 1e-4, 6 epoch, her epoch checkpoint.
- **Sonuç (aynı seed 2024):** kontrol v22 ep23 **%8** → v24 ep2/ep3 **%27**
  (+19 pp). Epoch kuyruğu: 21 / 27 / 27 / 18 / 14 / 17.
- Banette (100 maç): **%59** (v22 %55).
- Not: aynı modelin seed'ler arası oynaması büyük (v22: seed 2024 %8,
  seed 4242 %20) → 100 maçlık farklar gürültülü; kararlar 300 maçla verilecek.
- Script: `tools/mazlum/v24_run.ps1`.

### ⚠️ Metodoloji düzeltmesi (6 Kasım)

300 maçlık doğrulama, 100 maçlık taramaların yanıltıcı olduğunu gösterdi:

| vs Dragapult (300 maç, seed 777) | Galibiyet |
|---|---:|
| v22 ep23 (taban) | %14,3 |
| v24 ep2 | %15,0 |

Yani v24'ün "8 → 27" ekran farkı **seed gürültüsü**. Bundan sonra:
- Ekranlar 300 maç, karşılaştırmalar **aynı seed** ile.
- 100 maçlık farklara karar bağlanmaz (±5 pp).

### S1b — v25: karışık fine-tune (bitti, şüpheli)

- Veri: tam veri (168.720) + zor eşleşme (90.953) = **259.673 satır**
  (zor satırlar 2× ağırlıkta).
- v24 ep2'den, lr 5e-5, 4 epoch.
- Amaç: Dragapult kazancını korurken diğer eşleşmeleri kaybetmemek.
- Önce 300 maçlık doğrulama: v22 vs v24 ep2 (seed 777).
- Script: `tools/mazlum/v25_run.ps1`.

### S1c — v26: yalnız Slowking tarafı (çalışıyor)

- **Kök sorun hipotezi:** v22 verisinin yarısı rakibin kararları
  (168.720 satırın 85.950'si Slowking tarafı). Ajan hem Slowking'i hem
  rakip desteleri taklit ediyor → politika çakışması.
- v26 veri: 513 episode, **yalnız Slowking oyuncusunun kararları**
  (85.950 satır, 410 train / 103 val episode).
- Sıfırdan, 30 epoch, lr 1e-3; en iyi val ve son epoch 300 maçla
  Dragapult'a karşı (seed 777) test ediliyor. Taban: %14,3.
- Script: `tools/mazlum/v26_run.ps1`.

### Ladder uyarısı

- v22 aktif kaldıkça skor erimeye devam ediyor: 600,0 → **553,4** (sıra 414).
- v21 restore (939) kullanıcı onayı bekliyor; kota açık (6 Kasım).

### S2 — Veri genişletme
- Hedef: 800+ Slowking episode (şu an 513). Madencilik 429'a takılıyor;
  gece/aralıklı çalıştır.
- Öncelik: Alakazam (69) ve Dragapult (198); Alakazam insan galibiyeti %23.

### S3 — Kapasite karşılaştırması
- `MazlumNetBig` (296k, 4×4×80 Pre-LN) sıfırdan, 25 epoch, aynı veri.
- Banette'te kapasite aynayı bozmuştu; Slowking'de bağımsız denenecek.

### S4 — PPO-lite (CPU)
- 100-300k self-play oyunu (20 çekirdek ≈ 10 oyun/s → 3-8 saat).
- Lig: güncel + geçmiş checkpoint + `bots/dragapult` + `bots/banette`.
- λ=0.95, clip 0.2, entropy 0.01→0.001, 2-4 epoch.
- Gerekli: `selfplay_worker.py` + `ppo.py` + `league.py` (henüz yok).

### S5 — Uzmanlar + yönlendirme
- Dragapult/Alakazam/Grimmsnarl uzmanları; rakibin açılan kartlarına göre seçim.

## 5. Kapılar

- **Yerel test sadece legalite/çökme kapısı.** Seçim ladder A/B ile.
- Ladder'a çıkmadan önce: 100 maç Dragapult ≥ %35–40, Banette ≥ %50.
- Her submit öncesi: v21 (939) yedeği duruyor; A/B sonrası düşerse geri yükle.
- Günde 5 submission; önce restore, sonra deneme.

## 6. Devam checklist'i ("devam" denince)

1. `Get-Content $env:TEMP\mazlum_v24.log -Tail 30` → v24 sonucu.
2. Kota varsa ve kullanıcı onayladıysa **v21'i geri yükle**.
3. v24 kapıyı geçtiyse paketle (`--deck decks/lab/slowking_kangaskhan.csv`) ve
   A/B'ye gönder; geçmediyse S2/S3'e geç.
4. Push yok — kullanıcı onayı olmadan.

## 7. Komutlar

```powershell
# v24 eğitim + tarama
powershell -File tools/mazlum/v24_run.ps1

# karar izi ile teşhis
$env:MAZLUM_MODEL = "...\data\mazlum\v22_epochs\epoch_023.npz"
$env:MAZLUM_TRACE = "...\results\trace.jsonl"
python tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 50 --seed 7

# paket (model.npz'yi geçici değiştir, sonra v15'i geri koy)
python tools/build_submission.py --bot bots/mazlum --deck decks/lab/slowking_kangaskhan.csv --out submissions/mazlum_v25.tar.gz
```
