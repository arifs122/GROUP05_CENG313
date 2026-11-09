# MK / Ogerpon — Mega Kangaskhan + Teal Mask Ogerpon ajanı

Bu klasör, Top1 (`17014244`, `やる気元気ミワハルキ 2nd ver.`) listesini
kullanan bağımsız PTCG ajanının çalışma alanıdır. Diğer ajanların klasörlerine
ve mevcut `bots/` botlarına dokunulmaz; mevcut botlar sadece lokal rakip
olarak kullanılır.

## Deste

Top1'in kazandığı 50 replay'in step 0 submission action'ından çıkarıldı;
50/50 aynı liste. `deck_top1.csv` ve `bot/deck.csv` 60 karttır, engine
tarafından kabul edilir.

Ayrıntı: [`reports/deck.md`](reports/deck.md)
Kart davranışları: [`reports/card_behavior.md`](reports/card_behavior.md)

## Bot düzeni

`bot/` arena ve submission için hazır giriş noktasıdır:

| Dosya | Görev |
|---|---|
| `main.py` | Kaggle/arena giriş noktası; `agent(obs)` |
| `mk_state.py` | Gözlem yardımcıları (dict/dataclass uyumlu) |
| `mk_data.py` | Kart ID'leri, roller, prize değerleri, meta anahtarları |
| `mk_meta.py` | Search determinisation için public archetype listeleri |
| `mk_rules.py` | Katman 0/1: legal kural çekirdeği + enerji/prize/deck-out planı |
| `mk_search.py` | Katman 2: determinised 1-ply search + rollout |
| `mk_features.py` | Numpy ağları için state/option feature'ları |
| `mk_model.py` | Policy prior numpy inference |
| `mk_value.py` | Value net numpy inference |
| `mk_policy.py` | Orkestrasyon, süre koruması, legal fallback |

`mk_search.py` engine'in `search_begin/search_step/search_end` API'sini
kullanır; gizli kartlar bizim 60 kartlık listemizden ve public archetype
listelerinden örneklenir. Rakip deck listesi state feature'ı olarak
kullanılmaz.

Deck-out kontrolü sabit bir "deckCount <= 8" değildi; kalan prize ve tahmini
tur sayısına bağlıdır: `deck > max(2, 2*prize_left) + 4` güvenli kabul edilir,
altında isteğe bağlı draw yetenekleri/Lillie kapatılır.

## Çalıştırma

```bash
# tek maç
.venv/bin/python tools/arena.py \
  --bot0 agents/mk_ogrepon/bot \
  --bot1 bots/_template --games 10 --seed 7

# BO3
.venv/bin/python tools/arena.py \
  --bot0 agents/mk_ogrepon/bot \
  --bot1 bots/dragapult --games 10 --bo3 --seed 7

# 200 BO3, 5 paralel shard
.venv/bin/python agents/mk_ogrepon/tools/arena_parallel.py \
  --bot1 bots/dragapult --games 200 --workers 5 --seed 2026 \
  --out agents/mk_ogrepon/reports/bo3_200_dragapult_parallel.json

# search/prior kapatma
MK_SEARCH=0 MK_PRIOR=0 .venv/bin/python tools/arena.py ...
```

Varsayılanlar:

- `MK_SEARCH=1`, `MK_BUDGET=0.18` (karar başına search bütçesi)
- `MK_PRIOR=1`, `MK_PRIOR_WEIGHT=60`
- `MK_VALUE=1`, `MK_VALUE_WEIGHT=0` (value leaf katkısı kapalı)
- `MK_DECK` ile deck override edilebilir.

## Doğrulama / raporlar

- [`reports/robustness.md`](reports/robustness.md): 50 maç stress testi;
  **0 geçersiz aksiyon, 0 exception, 0 engine error, 0 fallback**;
  maksimum karar süresi ~41 ms (0.18 s bütçe). Search kapalı 150 maçta da 0 hata.
- [`reports/models.md`](reports/models.md): policy prior ve value net
  eğitim metrikleri + Torch↔numpy parite.
- [`reports/card_behavior.json`](reports/card_behavior.json): replay tabanlı
  kart davranış ölçümleri.
- [`reports/robustness.json`](reports/robustness.json): search açık makine çıktısı.
- [`reports/robustness_rules.json`](reports/robustness_rules.json): search kapalı 150 maç.
- [`reports/submission_notes.md`](reports/submission_notes.md): aşama/submission sırası.
- [`reports/bo3_200.md`](reports/bo3_200.md): 200 BO3 lokal arena sonuçları
  (template %44.5, Dragapult %9.0; 0 hata).

Yeniden üretmek için:

```bash
python3 agents/mk_ogrepon/tools/extract_deck.py
python3 agents/mk_ogrepon/tools/probe_cards.py
.venv/bin/python agents/mk_ogrepon/tools/build_dataset.py
.venv/bin/python agents/mk_ogrepon/tools/train_value.py
.venv/bin/python agents/mk_ogrepon/tools/train_prior.py
.venv/bin/python agents/mk_ogrepon/tools/stress_test.py 10
```

## Submission

Paket `tools/build_submission.py` ile üretilir (bu çalışmada **Kaggle'a
yüklenmedi**):

```bash
.venv/bin/python tools/build_submission.py --bot agents/mk_ogrepon/bot
```

Paket kökü: `main.py`, `deck.csv`, tüm `mk_*.py`, iki `.npz`, `cg/`.
Boyut ~2.25 MiB.

## Lokal sonuç notu

Lokal arena sadece crash/legal/süre kontrolüdür. Örnek 50 maçlık stress
testinde 26-24; farklı rakiplere göre değişir. Karar kriteri ladder'dır;
local kazanma oranı bu ajanın kabul kriteri değildir.
