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

### S1c — v26: yalnız Slowking tarafı (bitti: fark yok)

- 85.950 satır (513 episode, sadece Slowking oyuncusu), sıfırdan 30 epoch,
  val top-1 **%60,83**.
- **300 maç seed 777:** v26 ep30 **%14,7** · v22 taban **%14,3** → fark yok.
- Sonuç: iki taraflı veri kirliliği hipotezi de çürüdü.

### Veri kalitesi denetimi

- 513 Slowking episode'un **493'ü ilk-20 takımdan** (medyan sıra 2).
- Yani veri zaten elit; "kötü pilot" filtresi de çözüm değil.

### S1d — v27: kapasite testi (çalışıyor)

- `MazlumNetBig` (296k, 4×4×80 Pre-LN), v26 verisi (85.950 satır), sıfırdan,
  40 epoch, lr 1e-3.
- ep10 / ep25 / ep40, 300 maç Dragapult (seed 777). Taban %14,3.
- Script: `tools/mazlum/v27_run.ps1`.

### S1e — Genel güç karşılaştırması (bitti)

- Banette botuna karşı 300 maç (seed 777): v22 **%52,7** · v26 **%53,7**.
- Yani iki model de Banette'e karşı ~%53, Dragapult'a karşı ~%14.
- Tablo: Slowking politikası Banette seviyesinde oynuyor; sorun **Dragapult
  eşleşmesine özel** (insan Slowking pilotları Dragapult'a %44 kazanıyor).
- Not: %2 civarı `max_steps` draw var (Kaggle'da draw).

### İnsan replay denetimi (198 Slowking-vs-Dragapult maçı)

| | Kazanan (87) | Kaybeden (111) |
|---|---:|---:|
| Clefairy oynandı | %66 | **%83** |
| Clefairy saldırdı | %44 | %37 |
| **Smoochum saldırdı** | **%75** | **%75** |
| Kyurem saldırdı | %0 | %0 |
| Ortalama tur | 13 | 13 |

- Clefairy oynamak **kazanmayla ilişkili değil** (kaybedenlerde daha yüksek!).
- Asıl motor: **Smoochum "Delightful Kiss"** (2 Psychic Energy'yi yedeğe
  bağlar) — insanların %75'inde kullanılıyor, bizim ajanda %20.
- Slowking "Seek Inspiration" (213) bu enerjiyle Kyurem Trifrost / Clefairy
  Rondo kopyalıyor.
- **Kural testi:** Smoochum saldırısını zorla → **%13,0** (taban %14,7).
  Yine negatif: tek tek hamleleri zorlamak planı kurmuyor.
- Araç: `tools/mazlum/slowking_matchup_audit.py`.

### S1f — v28: kazanan maç ağırlıklandırma (çalışıyor)

- v26 verisi + Slowking'in Dragapult'u **yendiği** 87 maçın satırları 3×
  (12.006 satır → +24.012) = 109.962 satır.
- v26 ep30'dan, lr 5e-5, 6 epoch; ep6, 300 maç Dragapult (seed 777).
- Script: `tools/mazlum/v28_run.ps1`.

### Kayıp nedeni dağılımı (40 maç vs Dragapult botu)

Motorun `RESULT.reason` kodları (`vendor/cg/api.py`):
1 = 0 prize (tüm ödüller alındı) · 2 = deste bitti · 3 = aktifte Pokémon yok
(bench-out) · 4 = kart etkisi.

| reason | Maç | Yorum |
|---|---:|---|
| 1 (prize) | 34 | Normal prize yarışı kaybı |
| 3 (bench-out) | 6 | Yedek Pokémon bitmesi |

- Yani kayıpların **%85'i prize yarışı**, %15'i bench-out. Deste bitişi yok.
- Bench-out, Dragapult'un Trifrost yayılımına karşı korunmasız kalmaktan;
  tek başına düzeltilse bile kazanç ~5-7 pp ile sınırlı.

### S1g — v29: generalist → specialist (çalışıyor)

- Yazının tarifi: önce genelist, sonra deste uzmanı. Biz Slowking'i hep
  **sıfırdan** eğittik; bu ilk kez çok-deste genelistinden (`model_v14b_multi`)
  Slowking verisine fine-tune.
- 10 epoch, lr 1e-4; ep5 ve ep10, 300 maç Dragapult (seed 777). Taban %14,7.
- **Sonuç:** ep5 **%11,0** → ep10 **%17,7** (val hâlâ iyileşiyor: 61,64).
  İlk kez eğitim uzadıkça yükselen bir eğri var; 10 epoch erken kesmiş.
- Script: `tools/mazlum/v29_run.ps1`.

### S1h — v30: genelist→uzman devam (çalışıyor)

- v29 ep10'dan 20 epoch daha (lr 5e-5), ep5/10/15/20 × 300 maç Dragapult.
- Amaç: %17,7'nin üstüne çıkıyor mu, yoksa plato mu?
- Script: `tools/mazlum/v30_run.ps1`.

### Karar ağacı (kullanıcı dönünce)

1. **Restore onayı** → v21 (939) geri yükle.
2. v27 kapasite testi Dragapult'ta > %25 yaparsa → paketle, A/B'ye gönder.
3. Yapmazsa → BC tavanı kabul edilir; sıradaki gerçek adım **PPO-lite**
   (CPU, 100-300k oyun) veya özellik/encoder yükseltmesi.

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
