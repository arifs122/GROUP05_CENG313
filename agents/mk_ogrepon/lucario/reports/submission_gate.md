# Yeni submission kapısı — **HAZIR DEĞİL**

Tarih: 2026-11-03
Aday: `lucario_bc_pilot_v1` (rule core + BC prior)
Deste: `deck_real_petit.csv` (en iyi 2x2 hücresi)

## Hash

| Bileşen | SHA-256 (ilk 16) |
|---|---|
| `lucario_agent.py` | `aa299d7c68223740` |
| `lucario_features.py` | `cc234fad6bfb7190` |
| `lucario_model.py` | `5af85bf095df7e2d` |
| `policy_prior.npz` | `f31432d8dc9ab6cb` |
| `deck_real_petit.csv` | `1430e169017ec9e4` |
| `W` | `4116e394d1558d9b` |

## Yeni gate sonucu

| Metrik | Hedef | Sonuç |
|---|---:|---:|
| Dragapult proxy | ≥ 30% | **30-41.5%** (prior moduna göre) |
| External Lucario mirror | ≥ 40% | **20-27%** (en iyi argmax ~26%) |
| 0 crash / error | zorunlu | **0** (2x2 1600 maç) |
| 0 fallback | zorunlu | 0 gözlendi (2x2/BO1) |

**Gate FAIL: mirror < %40.** Dragapult tarafı geçiyor; mirror geçmiyor.
Bu yüzden submit yapılmadı.

## En iyi konfigürasyonlar

- Dragapult için: `deck_real_petit`, prior weight 120 → **%35.5**
- Mirror için: `deck_real_petit`, prior argmax → **~%26**
- İkisini aynı anda geçen konfigürasyon yok.

## Karar

- Deste/pilot 2x2: pilot açığı ~20 puan, deste açığı ~17-24 puan.
- Rule+BC prior MAIN uyumu **%58.4**; external referans **%66.9**;
  hedef %80.
- Ağırlık ayarına dönülmedi; `weights.json` yok.
- Dragapult planı (Solrock/Lunatone, Aura Jab, Dusk/Poké, Cosmic Beam)
  uygulandı ve Dragapult proxy %30+ bandına geldi; mirror hâlâ düşük.

## Submit için sıradaki iş

1. Mirror'u yükselt: external Lucario'nun erken Lunatone/Solrock + Aura Jab
   sıralamasını daha fazla replayden çıkar; BC prior eğitimini galibiyet +
   mağlubiyet tüm Mega Lucario episode'larına genişlet.
2. Hedef: MAIN top-1 ≥ %80; mirror ≥ %40; Dragapult ≥ %30.
3. Bu geçince submit onayı iste.
