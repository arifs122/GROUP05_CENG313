# Mazlum — Davranışsal Klonlama (Behavioral Cloning) Yol Haritası

Bu doküman, mevcut heuristic politikanın karar mantığını, elimizdeki replay
verisinin gerçek formatını ve **Mazlum** adını vereceğimiz öğrenilmiş politikayı
nasıl eğiteceğimizi özetler.

Kapsam: `src/agent/`, `bots/`, `benchmarks/`, `decks/`, `tools/replay_meta.py`.

---

## 1. Mevcut heuristic: kararlar nasıl veriliyor?

### 1.1 Baseline ajan — `src/agent/policy.py`

Tek fonksiyon: `act(obs) -> list[int]` (seçenek indeksleri). Mantık tamamen
**tip önceliği** üzerine kurulu, durum değerlendirmesi yok:

| Öncelik | OptionType | Not |
|---:|---|---|
| 0 | `EVOLVE` | her zaman ilk |
| 1 | `ABILITY` | |
| 2 | `ATTACH` | |
| 3 | `PLAY` | |
| 4 | `ATTACK` | |
| 5 | `RETREAT` | |
| 6 | `DISCARD` | |
| 7 | `END` | son çare |

- **Enerji bağlama:** `_best_main` = `min((öncelik, seçenek_indexi))`. `ATTACH`
  seçenekleri motor tarafından `(kart, hedef)` çifti olarak üretildiği için
  pratikte **ilk yasal enerji, ilk yasal hedefe** (genelde aktif Pokémon) gider.
  Enerji türü, hedefin rolü, saldırı maliyeti, `energyAttached` bayrağı
  **hiç dikkate alınmaz**.
- **Saldırı seçimi:** `_best_attack` = `all_attack()` tablosundaki **basılı
  hasar** maksimumu (`damage`, eşitlikte küçük indeks). Zayıflık, direnç,
  hedefin kalan HP'si, KO potansiyeli, enerji maliyeti yok → aynı saldırı
  her durumda seçilir.
- `YES_NO` → her zaman `YES`. `COUNT` → draw/damage-counter bağlamlarında
  maksimum, diğerlerinde minimum. Kart seçimlerinde discard bağlamı
  `minCount`, diğerleri `maxCount` adet **ilk N seçenek**.
- Deck: `PTCG_DECK` → `deck.csv` → `decks/current.csv` →
  `decks/mega_abomasnow.csv` sırasıyla ilk bulunan 60 kartlık dosya.

Bu politika "yasal hamle üret" garantisi verir, oyun kalitesi hedeflemez.

### 1.2 Takım botu — `bots/banette/main.py`

Skor tabanlı, elle ayarlanmış ağırlıklar (`_main_score`):

- **ATTACH skoru:** enerji türü tabanı — `Telepath 4200`, `Basic Psychic 3800`,
  `Legacy 3200`, `Air Balloon 4000` (aktif enerjisiz + bench'te hazır
  Dhelmise/Banette varsa), diğer `1800`. Hedef bonusu: `Dhelmise/Banette +600`,
  `Bloodmoon/Lillies Clefairy +100`. **Aktif Pokémon'un hiç enerjisi yoksa
  `+2500`** — yani "önce aktifini aç" kuralı.
- **Saldırı skoru:** `_attack_damage` durum hesabı yapar (Vengeful Anchor
  30+140, Blood Moon 240, Full Moon Rondo 20+20×(iki bench), Matcha Spin
  spread, zayıflık ×2, Dragon/Psychic özel durumu). Sonra:
  `KO ise 10000 + prize_value×1000`, değilse `hasar×10 + 100` + saldırıya
  özel bonuslar (Puppet Pull +500, Blood Moon +300, Hex Hurl +200).
- Diğer bağlamlar (setup, discard, search, switch) ayrı skor tabloları;
  `_pick_top` eşik altındaki hamleleri eler (`max ≤ 0` ise hiç oynamaz).

### 1.3 Referans ajanlar — `benchmarks/`

- `meta_decks/_policy.py`: config'den gelen skorlar. Attach: enerji kartı
  `4500`, hedef listesinde `+800`, boş aktif `+1200`. Attack: KO `12000`,
  değilse `1000 + hasar×10 + bonus`.
- `external_agents/*` (wmh/ptcg-abc): aynı skorlama ailesi, deck'e özel
  ağırlıklarla; `bellibolt/value_weights.py` ise **eğitilmiş** 22→16→1 board
  value MLP'si (win prob). Yani repoda öğrenilmiş model fikri zaten var,
  sadece politika tarafı eksik.

**Ortak zayıflık:** tüm ağırlıklar elle; durum temsili (state embedding) yok,
plan/archetype modeli yok, kararlar birbirinden bağımsız. Mazlum'un kazanç
alanı tam burası: insan/uzman kararlarını taklit eden, duruma koşullu bir
politika.

---

## 2. Replay verisi: elimizde ne var, formatı ne?

### 2.1 `decks/` içinde ham replay yok

`decks/` altındaki "replay-mined" çıktı **türetilmiş** veridir:

| Yol | Format | İçerik |
|---|---|---|
| `decks/meta/*.csv`, `decks/lab/*.csv` | CSV, tek sütun, 60 satır | kart ID listesi |
| `decks/meta/README.md` | Markdown | 188 episode / 376 deck agregası, archetype dağılımı, W-L tablosu |
| `decks/meta/matchups.md` | Markdown | matchup notları |

Yani ham `observation → action` çiftleri repoda **tutulmuyor** (lisans +
`.gitignore`). `replays/` klasörü git dışı ve şu an boştu.

### 2.2 Ham replay nasıl alınır

```bash
python -m kaggle competitions submissions -c the-pokemon-company-ptcg-ai-battle-challenge-playground
python -m kaggle competitions episodes <submission_id>
python -m kaggle competitions replay <episode_id> -p replays
```

Doğrulama için bir replay indirildi: `replays/episode-125199547-replay.json`
(13.2 MB, 481 step, Dragapult vs Mega Kangaskhan).

### 2.3 Şema (doğrulanmış)

```
{
  "id", "name", "version", "schema_version", "module_version",
  "configuration", "specification", "description", "title",
  "info": {"EpisodeId", "TeamNames": [..2], "Agents": [{Name}, {Name}]},
  "rewards": [1, -1],          # kazanan = 1, kaybeden = -1, beraberlikte eşit
  "statuses": ["DONE", "DONE"],
  "steps": [                    # steps[t] = t anındaki 2 oyuncunun durumu
    [ {"action", "info", "observation", "reward", "status", "visualize"},
      {...} ],
    ...
  ]
}
```

`observation` alanları:

```
select: {type, context, minCount, maxCount, option[], deck, contextCard,
         effect, remainDamageCounter, remainEnergyCost}   # deck seçiminde null
current: {turn, turnActionCount, firstPlayer, yourIndex, supporterPlayed,
          stadiumPlayed, energyAttached, retreated, result, stadium, looking,
          players[2]: {active[Pokemon], bench[], deckCount, discard[], prize[],
                       handCount, hand, poisoned, burned, asleep, paralyzed,
                       confused, benchMax, win}}
logs: [...]
search_begin_input
```

`option` alanları: `type` (PLAY/ATTACH/EVOLVE/ABILITY/ATTACK/RETREAT/END/
NUMBER/CARD/…), `index`, `area`, `inPlayArea`, `inPlayIndex`, `attackId`,
`number`, `serial`, `count`, `specialConditionType`.

**Kritik ayrıntılar**

1. **Action eşlemesi:** `steps[i]` gözlemi, `steps[i+1][player]["action"]`
   ile cevaplanır (`tools/replay_meta.py: iter_decisions`). Yanlış eşleme
   tüm veri setini bozar.
2. **Deck listesi:** `steps[0][0].visualize[0].action` = `[deck0, deck1]`.
3. **Kısmi gözlem:** rakip eli `None` (sadece `handCount`), prize kartları
   `None`. Mazlum da aynı bilgiyle oynayacağı için bu doğal; ama "rakip elde
   ne var" özelliği **üretilemez**.
4. `select.type` ile `select.context` ayrı şeylerdir: `MAIN` (type 0) tek
   başına kararların ~%52'si; kalanı hedefli seçimler (to_hand, damage counter,
   attach_to, switch, discard…).

### 2.4 Tek replay'den ölçülen istatistikler

- 479 karar (481 step), **ortalama 5.9 seçenek/karar**, maksimum 33.
- Context dağılımı: MAIN 251, TO_HAND 70, DAMAGE_COUNTER_ANY 54, ATTACH_TO 30,
  SWITCH 17, TO_ACTIVE 13, TO_BENCH 8, DISCARD_ENERGY 7, ATTACH_FROM 7,
  SETUP_ACTIVE 6, SETUP_BENCH 4, DISCARD 4, EVOLVE 4, IS_FIRST 3, ACTIVATE 1.
- MAIN etiketleri: PLAY 98, ABILITY 52, ATTACH 35, ATTACK 34, EVOLVE 12,
  RETREAT 11, END 9.
- Attach hedefleri: 18 aktif, 17 bench → enerji kararı gerçekten anlamlı bir
  sınıflandırma problemi (heuristic bunu yalnızca "ilk seçenek" ile çözüyor).

**Hacim tahmini:** ~480 karar/maç → 1.000 replay ≈ 480 bin örnek, 5.000
replay ≈ 2.4 milyon. Ham replay ~13 MB/maç olduğu için ham dosyalar işlendikten
sonra silinmeli; kalıcı format sıkıştırılmış tablo olmalı.

---

## 3. Mazlum — BC yol haritası

### Hedef deste

`decks/meta/README.md` (188 episode / 376 deck) içindeki en yüksek galibiyet
oranı **Banette**: 57 deck, 36-21, **%63.2**. Bu yüzden Mazlum'un hedef destesi
`decks/meta/banette.csv` (consensus liste) ve eğitim verisi yalnızca Banette'in
kazandığı maçlardan toplanır (`tools/mazlum/collect_replays.py`).

### Mevcut durum (veri seti v5 — taktiksel)

- 300 episode / 108.219 karar, %80/%20 episode split (86.172 / 22.047 satır).
- **State 73 boyut:** aktif HP, prize farkı, el sayısı, iki aktif Pokémon ID'si,
  **rakip aktif HP'si**, **rakip aktif enerji sayısı**, **iki tarafın bench doluluk
  oranı** + **64'lük deste histogramı**.
- **Option 41 boyut:** 40 mevcut özellik + **`attack_lethal`** bayrağı
  (basılı hasar + zayıflık çarpanı rakibin HP'sini aşıyorsa 1).
- **Model:** gizli katman 64 → **128** (154.625 parametre), embedding 2048×32.
- `train_bc.py`: early stop epoch 19, en iyi **epoch 9**; train top-1 %56.10,
  **val top-1 %52.51**, val loss 1.3114, gap +3.59 pp.
- Arena, **300 tekli maç**: **%79.7** (239-61), 0 hata, %95 GA ~%75-84.
  Önceki: v4 %74.0 (300 tekli), v3 %75.0 (100 tekli).

### v6 — DAgger turu 1 (düzeltilmiş)

İlk tur bozuktu (`%38.3`): uzman kendi Kieran listesiyle etiketliyordu, satırlar
ağırlıksızdı ve model val top-1 ile seçiliyordu. Düzeltmeler:

- **Uzman senkronizasyonu:** her iki bot da aynı 60 kartı (consensus Banette)
  oynuyor; 100 maç → **7.713 satır** (toplama sırasında Mazlum 62-38 önde).
- **Örnek ağırlığı:** replay 1.0, DAgger **0.3** (ağırlıklı cross-entropy).
- **Erken durdurma:** her 5 epoch'ta **30 maçlık mini-arena**, en iyi model
  galibiyet oranına göre (`patience=2` değerlendirme).
- Eğitim: best **epoch 5** (mini-arena %73.3), train %52.97, val %50.19;
  epoch 15'te durdu (mini-arena %66.7 → %46.7).
- Arena 300 tekli: **%67.0** (201-99), 0 hata.
- Değerlendirme: v5 %79.7 > v6 %67.0 > bozuk v6 %38.3. 30 maçlık mini-arena
  seçimi gürültülü (30 maç ≈ ±16 pp); final 300 maç %67 çıktı.
- v5 ağırlıkları `data/mazlum/model_v5.npz` olarak saklandı; geri dönmek için
  `Copy-Item data/mazlum/model_v5.npz data/mazlum/model.npz`.

### v6 — DAgger turu 2 (300 maç, 100'lük mini-arena)

- Mini-arena 30 → **100 maç** (her 5 epoch, patience 2) → seçim artık final
  testle tutarlı (mini %46 → final %41, 30 maçlık turda %73 → %67 sapması vardı).
- Uzman senkronizasyonu korundu: iki bot da consensus Banette (60 kart).
- Toplama: 300 maç → **23.567 satır** (toplama sırasında Mazlum 221-79 önde,
  yani **uzman öğrenciden zayıf**).
- Dataset: 108.219 → **131.786 satır**; DAgger ağırlığı 0.3.
- Eğitim: best epoch 10 (mini-arena %46), train %57.81, val %52.04;
  epoch 20'de durdu (mini-arena %35 → %36).
- Arena 300 tekli: **%41.0** (123-177), 0 hata.
- Teşhis: v6 yine daha az saldırıyor (10 maçta ATTACK 54 vs v5 74, END 59 vs 29).
  Kök neden veri değil **öğretmen**: Banette heuristiği consensus desteyle
  oynarken v5'ten zayıf; DAgger zayıf öğretmeni taklit ederek aşağı çekiyor.
- Sonuç sıralaması: v5 %79.7 > v6-tur1 %67.0 > v6-tur2 %41.0.
  Üretim için v5'te kalınmalı.

### v7 — Devasa BC (1.133 episode)

- Toplama: **1.133 Banette galibiyeti**. 2.000 hedefi Kaggle API'sinin 429
  (Too Many Requests) limitine takıldı; 194 submission tarandı, 89 probe 429
  yedi. `collect_replays.py`'ye retry/backoff eklendi (gelecek turlar için).
- Dataset: **436.204 karar / 1.133 episode** (227 val episode), 17,7 MB.
- Eğitim: best **epoch 10** (mini-arena 87/100), train %60.18,
  **val top-1 %55.17**, epoch 20'de erken durdu.
- Arena 300 tekli: **%74.0** (222-78); bağımsız ikinci 300 maçlık koşu **%76.0**
  (228-72) → **birleşik %75.0 (450/600)**. Aynı ağırlıkların in-process 100
  maçlık replikası 80/100 (motor RNG'si sabit olmadığı için ölçümler ±5-7 pp
  salınıyor).
- Yorum: veri 3.8× büyüdü, val top-1 +2.66 pp arttı (%52.51 → %55.17) ama
  v5'e karşı galibiyet oranı istatistiksel olarak aynı kaldı (%74.0 vs %79.7;
  %95 GA'lar örtüşüyor). Model mimarisi ve `model.npz` boyutu (≈566 KB) sabit —
  Kaggle gönderim sınırı açısından değişiklik yok.

### v8 — Attention + hibrit lethal filtresi

- **Mimari:** state → query, options(+kart embedding) → key/value; tek başlı
  **dot-product attention** (d=64), maskeli softmax, context + token → 128'lik
  ReLU head. Parametre **106.177 (~415 KB)**, `model.npz` **389.5 KB**
  (< 600 KB hedefi).
- **NumPy ↔ PyTorch:** max fark 7.5e-08, argmax uyumu %100.
- **Hibrit politika:** yasal `attack_lethal == 1` seçenek varsa ağ logitleri
  tamamen ezilir. Ablasyon (100 maç): filtreli 74/100, filtresiz 74/100,
  100 maçta 121 override → bu eşleşmede nötr (model zaten lethal'i seçiyordu).
- **Eğitim:** best epoch 20, mini-arena %89, train %61.95, **val %57.46**
  (tüm sürümlerin en iyisi), `patience` ile epoch 30'da durdu.
- **Arena 300 tekli:** **%79.3** (238-62), 0 hata.
- Sonuç: v8, v5 ile aynı galibiyet oranında (%79.3 vs %79.7) ama **%31 daha
  küçük model** ve belirgin biçimde daha iyi val top-1 (%57.46 vs %52.51).

### v9 — 2 başlı attention × 2 blok, 1.000 maç ölçümü

- **Mimari:** 2 head × 32 dim, iki attention bloğu (state query → blok 1,
  `[state, context1]` query → blok 2), arada residual FFN; 192→128→1 head.
  **147.521 parametre (~576 KB raw)**, `model.npz` **541,4 KB** (< 600 KB).
- **NumPy ↔ PyTorch:** max fark 7,45e-08, argmax %100. Lethal filtresi korundu.
- **Veri:** 1.133 episode / 436.204 karar. Archetype çeşitlendirmesi Kaggle
  429 limitine takıldı (746 submission tarandı, 204 probe 429; 0 yeni).
  Buna karşın veri **12 archetype / 2.266 deste** içeriyor (her maçın iki
  oyuncusunun kararları da sette).
- **Eğitim:** best epoch 5 (mini-arena 88/100), train %61.20, val %57.32;
  epoch 15'te durdu (arena 88/88/88 → eşitlik ilk modeli korudu).
- **Arena 1.000 tekli:** **%81.8** (818-182), 0 hata, %95 GA ±2,4 pp.
- Sonuç: v9, 1.000 maçlık en güvenilir ölçümde v8/v5'in biraz üzerinde
  (81,8 vs 79,3/79,7) ve GA'lar örtüşse de üst sınırı zorluyor.

### Faz 0 — Replay madenciliği (`tools/mazlum/collect_replays.py`)

- Leaderboard'dan ilk N takım → `team-submissions` → `episodes` → `replay`.
- Filtreler: tamamlanmış episode, `rewards` kesin (beraberliği at veya
  ayrı işaretle), bozuk/eksik step at, `episode_id` ile dedupe.
- Etiket ağırlığı: **kazananın kararları 1.0, kaybedenin 0.5** (opsiyonel;
  sadece kazananları almak da meşru bir başlangıç).
- Rate limit + retry; indirilen dosya işlenir işlenmez silinir.

### Faz 1 — Veri seti (`tools/mazlum/build_dataset.py`)

Her karar bir satır:

```
state_features        float32[D]
option_features       float32[N, F]   # N = o anki yasal seçenek sayısı
option_mask           bool[N]
label                 int64           # seçilen indeks (tekil seçimler)
label_multi           bool[N]         # minCount/maxCount > 1 ise
meta: episode_id, player, deck_archetype, turn, select_type, select_context
```

- Çoklu seçimler (`maxCount > 1`): iki seçenek — (a) çoklu-etiket BCE,
  (b) seçimleri sırayla autoregressive üret (daha doğru, daha pahalı).
  İlk sprintte (a) yeterli.
- Split **episode bazında** (aynı maçın kararları train/val'a dağılmasın).
- Çıktı: `data/mazlum/dataset.parquet` (git dışı) + `stats.json`.

### Faz 2 — Özellikler (`src/mazlum/features.py`, tek kaynak)

Uygulandı: `state` **73 boyut**, `options` **41 boyut** (40 satır padding) +
`card_ids` **int16 [40, 2]**.

- **State (73):** aktif HP, prize farkı, el sayısı, **kendi/rakip aktif Pokémon
  ID'si**, **rakip aktif HP'si**, **rakip aktif enerji sayısı**, **iki tarafın
  bench doluluk oranı** + **64'lük deste histogramı**
  (`card_id % 64` sayımı / 60 → deste-koşullu politika).
- **Option (41):** option type one-hot (17), hedef bayrakları (4), normalize
  `index/area/number/attackId` (5), **saldırı taban hasarı**, **enerji
  gereksinimi** (adet + 12 enerji tipi sayımı), **`attack_lethal`** (bu saldırı
  rakibi bu tur bayıltıyor mu: basılı hasar + zayıflık ×2 ≥ rakip HP).
- **Model:** gizli katmanlar **128** (State/option/scorer MLP), embedding 2048×32.
- **Kart kimliği:** `card_ids` (hamle yapan + hedef) → **`nn.Embedding`
  (2048 × 32, padding 0)**; tablo `card_embedding.weight` olarak `model.npz`'de
  ve saf NumPy çıkarımında lookup table olarak kullanılıyor.
- Girdiler eğitimde z-score ile standardize edilir; `state_mean/std` ve
  `option_mean/std` `model.npz` içinde saklanır (çıkarım aynı dönüşümü uygular).
- Featurizer hem eğitimde hem çıkarımda **aynı** kod olmalı; aksi halde
  train/serve kayması olur.

### Faz 3 — Model

- **Pointer/ranking ağı:** state encoder (MLP 256→256) + option encoder,
  her seçenek için skor → maskeli softmax → seçilen indekste NLL. Değişken
  seçenek sayısını doğal çözer ve **yasal olmayan hamle üretemez**.
- Alternatif: küçük transformer (state + option token'ları), ama 2.4M örnek
  ve CPU kısıtı düşünülürse MLP yeterli.
- Boyut hedefi: < 5 MB, karar başına < 10 ms (Kaggle 600 s overage bütçesi).
- İkinci kafa (opsiyonel): value head (win prob) → sonraki fazda self-play
  fine-tune için.

### Faz 4 — Eğitim (`tools/mazlum/train_bc.py`)

- Kayıp: masked cross-entropy (tekil) + BCE (çoklu).
- Metrikler: top-1, top-3, legal-rate (%100 olmalı), ECE/kalibrasyon,
  bağlam bazında doğruluk (MAIN / attach / discard ayrı — attach doğruluğu
  kritik).
- Sınıf dengesizliği: PLAY/ABILITY baskın; bağlam bazlı örnekleme veya
  ağırlıklandırma.
- Deck koşullama: her bot tek deck kullandığı için önce **deck sabit**
  (ör. Banette) eğitip sonra deck embedding ile genelleştir.

### Faz 5 — Değerlendirme (`tools/mazlum/eval_bc.py` + mevcut `tools/arena.py`)

- `python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 100 --bo3`
- Sabit deck ligi (`--deck decks/meta/banette_kieran_bloodmoon_neutral.csv`)
  ile yalnızca politika karşılaştırılır.
- Referans: `benchmarks/external_agents/*` ve `benchmarks/meta_decks/*`.
- Uyarı: **taklit doğruluğu ≠ kazanma oranı.** Karar doğruluğu yükselirken
  win-rate düşebilir; karar eşiği her zaman arena sonucudur.

### Faz 6 — Paketleme (`bots/mazlum/`)

`bots/README.md` sözleşmesine uy:

```
bots/mazlum/
  main.py            # def agent(obs_dict) -> list[int]
  mazlum_policy.py   # benzersiz yardımcı modül adı (kural gereği)
  model.npz          # ağırlıklar, saf numpy
  deck.csv           # 60 kart
  meta.json          # {"name": "mazlum", "author": ..., "notes": ...}
```

- Kaggle paketinde **torch yok**; çıkarım saf numpy ile yazılmalı
  (veya ağırlıklar `.npz` + elle matmul). Eğitim tarafı serbest.
- `python tools/build_submission.py --bot bots/mazlum` ile `submissions/`
  altına paketle (git dışı).

### Faz 7 — İyileştirme döngüsü

1. **DAgger:** Mazlum oynar, heuristic/uzman etiketler; yeni veri eklenir.
2. **Self-play:** Mazlum vs Mazlum maçlarından veri (yalnızca yüksek
   kaliteli/kazanan taraflar).
3. **Veri ağırlıklandırma:** kazanan, yüksek ladder, uzun maçlar.
4. **Archetype koşullama:** rakip deck tahmini (archetype classifier zaten
   `replay_meta.archetype` içinde var) → matchup'a özel politika.

---

## 4. İlk sprint (kabul kriterleri)

| # | İş | Çıktı | Kabul |
|---:|---|---|---|
| 1 | `collect_replays.py` | `replays/*.json` | 100 episode indirilebilir, dedupe çalışıyor |
| 2 | `build_dataset.py` | `data/mazlum/dataset.parquet` | ~48k karar, action eşlemesi doğrulanmış |
| 3 | `features.py` + `train_bc.py` | `model.npz` | val top-1 ≥ %60, legal-rate %100 |
| 4 | `bots/mazlum` | arena sonucu | 100 BO3 maçta `bots/banette`'e karşı ≥ %40 |
| 5 | Dokümantasyon | `docs/mazlum-bc-roadmap.md` güncel | sayılar gerçek veriyle eşleşiyor |

> Not: 188 episode'lık mevcut örneklem (bkz. `decks/meta/README.md`) **yanlı**:
> leaderboard ilk 15 takım, takım başına ilk 8 episode. Mazlum için veri
> toplarken takım/episode çeşitliliğini artırın; aksi halde model birkaç
> takımın stilini kopyalar.

---

## 5. Riskler

- **Kısmi gözlem:** rakip eli/prize kartları gizli → özellik olarak yok;
  model bunları "tahmin etmeye" çalışmamalı.
- **Veri sızıntısı:** aynı episode'un farklı kararları train ve val'a
  dağılırsa metrikler şişer (episode bazlı split zorunlu).
- **Kopya stil:** BC, toplanan ajanların hatalarını da öğrenir; DAgger ve
  arena filtresi şart.
- **Lisans:** replay'ler ve deck listeleri competition-use only; `replays/`,
  `data/` ve üretilen veri setleri repoya commit edilmez (`.gitignore`).
