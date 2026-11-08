# Sabah raporu — 8 Kasım 2026

## 1. Takım durumu

| | |
|---|---|
| Takım skoru | **861,6** · sıra **221** |
| Aktif son 2 | `shaco ppo` (58097253) + `gragas ppo` (58097239) — arkadaşın |
| Bizim son gönderimiz | Dipam v3 (58093588) — %47,4 · sıra 323 döneminde |
| Kota | Sabah sıfırlanır |

**Önemli:** Arkadaşın PPO modelleri skoru 689 → 862'ye taşıdı. Bizim adaylarımız bu seviyeyi geçmeden submit edilirse, son-2 takibinden bir PPO modeli düşer ve skor geriler. Üstelik o modellerin dosyası bizde yok; geri yükleyemeyiz.

## 2. Gece yapılanlar

### Round-robin (20 maç/eşleşme, seed 777)

| Aday | vs Dragapult | vs Alakazam | vs Archaludon | Ort. |
|---|---:|---:|---:|---:|
| **public Grimmsnarl** | %65 | %35 | %50 | **%50** |
| **public Alakazam** | %40 | — | %60 | **%50** |
| public Archaludon | %50 | %30 | — | %40 |
| v41 Banette | %50 | %40 | %15 | %35 |
| v15 Banette | %45 | %35 | %10 | %30 |
| v21 Banette | %35 | %30 | %5 | %23 |
| Dipam v3 | %0 | %0 | %0 | bozuk (yerel) |

> Not: v21 yerelde en kötü ama ladder'da en iyi bilinen ajanımız (%50). Yerel sıralama ladder'ı öngörmüyor.

### v43 Banette (gece eğitimi)

- 700.000 satır · 1.775 episode · feature v2 · big arch · 12 epoch
- Val top-1 %61,3

| Checkpoint | Dragapult | Alakazam | Archaludon |
|---|---:|---:|---:|
| ep6 | %53 | %52 | %9 |
| **ep12** | **%60** | %39 | %6 |

**Sonuç:** Dragapult'ta güçlü, Alakazam'da orta, **Archaludon'da hâlâ çöküyor** (yapısal deste sorunu).

## 3. Push'a hazır paketler

| Dosya | İçerik | Boyut |
|---|---|---:|
| `submissions/public_grimmsnarl.tar.gz` | public Grimmsnarl (arama ajanı) | 2,04 MiB |
| `submissions/public_alakazam.tar.gz` | public Alakazam kural ajanı | 2,06 MiB |
| `submissions/public_meta.tar.gz` | public Archaludon ajanı | 2,05 MiB |
| `submissions/mazlum_v43_banette.tar.gz` | v43 ep12 (big + v2) | 3,11 MiB |
| `submissions/mazlum_v21_matchups.tar.gz` | v21 (ladder kanıtlı ~%50) | 2,56 MiB |

Hepsi yerelde çalışıyor (0 hata), cg dahil.

## 4. Öneri

1. **Şimdi submit etme.** Arkadaşın PPO'ları 862'de; bizim adaylar bunu geçecek kanıta sahip değil, üstelik geri dönüş dosyası yok.
2. Deneyeceksen **tek A/B**: `public_grimmsnarl` (en dengeli local profil) — ama önce arkadaşın `gragas/shaco` dosyasını al ki düşerse geri koyabilelim.
3. 1k hedefi için asıl yol: **public 983,8 ajanını birebir çıkarmak** (model ağırlıkları elimizde ve hash doğrulandı; eksik olan main.py+deste paketi) veya kendi RL hattımız.

## 5. Teknik notlar

- Feature v2 (47/44) ve şema otomatik algılama çalışıyor (v1 modeller bozulmuyor).
- Public rakip havuzu: Alakazam, Grimmsnarl, Archaludon (+ bizim Dragapult/Banette/Lucario).
- Veri: 8.153 episode manifest; Slowking 1.929, Dragapult 5.910, Banette 3.603.
