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

### v10 — 1.800 episode, düzeltilmiş erken durdurma

- **Veri:** 1.800 Banette/meta galibiyeti (27 archetype), **679.330 karar**
  (545.466 train / 133.864 val, 360 val episode).
- **Erken durdurma:** `>=` + eşit arena skorunda **val top-1 tie-break**.
- **Eğitim:** best **epoch 25** (arena 90/100), train %60,10, val %56,45;
  epoch 35'te durdu.
- **Arena 1.000 tekli:** **%85.6** (856-144), 0 hata, GA ±2,2 pp.
- Aynı düzeltmeyle 1.133 episode'luk sette 1.000 maç **%87.8** çıkmıştı;
  fark 2,2 pp (~2σ) ve veri çeşitliliğinin Banette matchup'ını seyreltmesiyle
  açıklanabilir. Mimari ve `model.npz` boyutu (541 KB) sabit.

### v11 — "Katilini Klonla" (anti-meta)

- Kaggle'daki resmi **Mazlum submission'ı** (ref 57643520) bulundu; 91 episode
  indirildi, 53 mağlubiyet / 35 galibiyet; **2 mirror maç atlandı** (iki deste
  aynı olduğu için taraf belirsiz).
- `parse_episode(players={winner})` ile **yalnızca kazananın kararları**
  çıkarıldı: **8.683 satır**, `source=2`, **ağırlık 2.0**.
- Dataset: 679.330 → **688.013 satır** (validasyon split'i değişmedi).
- Eğitim: best **epoch 10** (arena 86/100), train %60.77, **val %56.74**;
  epoch 20'de durdu.
- Arena 1.000 tekli: **%87.4** (874-126), 0 hata.
- Not: ilk koşu (mirror hatası düzeltilmeden) %80.8 vermişti; **etiketleme
  hatası giderilince 6,6 pp kazanç** ve v10'un (85,6) üzerine +1,8 pp.

### v12 — FP16 + 4 blok × 4 baş (regresyon)

- **FP16 quantization:** `save_numpy` tüm dizileri float16 yazıyor (61 dizi),
  `agent.py` yüklemede float32'ye upcast ediyor. `model.npz` **551,3 KB**
  (< 600 KB), fp32 v11 (541 KB) ile aynı bütçede **2× parametre**.
- **Mimari:** d=80, 4 baş × 20, **4 blok** (q/k/v/o + residual FFN 80→144→80),
  head 160→128→1. **297.409 parametre** (~581 KiB raw).
- **NumPy ↔ PyTorch:** max fark 1,24e-4, argmax %99,5 (fp16 yuvarlaması).
- **Eğitim:** best epoch 15, train **%50,96**, val **%50,14**, loss 1.354
  (v11: %60,77 / %56,74). Derin residual yığını **LayerNorm olmadan**
  optimize olamadı; eğitim ~9 dk/epoch sürdü.
- **Arena 1.000 tekli:** **%76,4** (764-236), 0 hata.
- Sonuç: FP16 hedefi tuttu (2× parametre, aynı dosya boyutu), fakat 4 bloklu
  mimari bu haliyle v11'in (%87,4) altında. Üretim paketi v11'de bırakıldı.

### v13 — Dragapult "Beyin Nakli"

- **Rollback:** v12 ağırlıkları Git'te yok (yalnızca v9 commit'li); v11
  ağırlıkları `submissions/mazlum.tar.gz` içinden geri yüklendi (147.521
  parametre, 100 maç doğrulaması %87,0). v12 kodu (4 blok, FP16) temizlendi.
- **Veri:** 245 Dragapult galibiyeti (Dragapult'un kendi hamleleri) + 348
  Dragapult mağlubiyetinde rakibin (anti-meta) hamleleri → **593 episode /
  104.222 satır** (`dataset_dragapult.npz`, 3,75 MB).
- **Eğitim (v11 mimarisi, 2 blok × 2 baş, FP32):** best **epoch 40**
  (mini-arena 67/100), train %65,66, **val %55,45**.
- **Arena 1.000 tekli** (Dragapult destesi, aynı Banette botu): **%58,9**
  (589-411), 0 hata.
- Kıyas: Banette beyni %87,4 (Banette destesi) — fark hem deste matchup'ından
  hem veri hacminden (104k vs 688k satır) kaynaklanıyor. Dragapult destesi
  Banette botuna karşı yapısal olarak daha zor bir eşleşme.

### v12.1 — 4 blok × 4 baş + Pre-LayerNorm (Dragapult)

- **Onarım:** her bloğa Pre-LN eklendi (`ln_q` sorgu, `ln_1` token'lar,
  `ln_2` FFN öncesi; `eps=1e-5`). v12'de eğitilemeyen derin residual yığını
  artık öğreniyor: train top-1 %50,96 → **%67,57**.
- **NumPy:** `_layer_norm` float32 `(x-mean)/sqrt(var+eps)*w+b`; 4 bloklu
  forward. FP16 ağırlıklar yüklemede float32'ye upcast.
- **Boyut:** **299.329 parametre**, `model.npz` **559,3 KB** (fp16, 85 dizi);
  NumPy ↔ PyTorch max fark 1,63e-4.
- **Eğitim (Dragapult verisi 104.222 satır):** best **epoch 25**
  (mini-arena 68/100), train %67,57, **val %56,15**.
- **Arena 1.000 tekli** (Dragapult destesi, Banette botu): **%68,2**
  (682-318), 0 hata → v13'e göre **+9,3 pp**.
- Durum: `model.npz` = v12.1 Dragapult, `deck.csv` = Dragapult; Banette
  ağırlıkları `model_v11_banette.npz` olarak saklandı.

### v12.2 — Dragapult veri ölçekleme (1.800 episode)

- **Toplama:** 593 → **1.800 Dragapult episode** (71 dk, 0 probe hatası;
  `collect_dragapult.py` Dragapult submission'larının tüm maçlarını indirir,
  kazanan tarafı kaydeder). Havuz: 1.141 Dragapult galibiyeti / 659 mağlubiyet.
- **Dataset:** replay'lerden doğrudan **385.435 satır / 1.788 episode**
  (307.674 train / 77.761 val, 358 val episode).
- **Eğitim:** erken durdurma epoch 15'te; seçilen **epoch 5** (mini-arena
  62/100), train %54,95, val %54,81.
- **Arena 1.000:** **%51,9** (519-481) — 104k satırlık v12.1'in (%68,2)
  altında.
- **Teşhis:** mini-arena skorları 62 → 52 → 61 (gürültü); `patience=2` ile
  epoch 5 seçildi, oysa val epoch 15'te %56,98'e çıkıyordu. Yani veri
  ölçekleme değil, **gürültülü arena seçimi** modeli erken kesti.
- En iyi ağırlık (68,2%) paketten geri yüklendi; `model.npz` = %68,2 sürümü.
- Öneri: aynı veriyle `--patience 4-5` veya 300 maçlık mini-arena ile seçim.

### v12.3 — Erken durdurma onarımı (385k, patience 5)

- `patience` **2 → 5**, mini-arena **100 → 200 maç** (`>=` + val top-1
  tie-break korundu); 30 epoch sınırı.
- **Eğitim (385.435 satır):** best **epoch 20** (mini-arena 147/200 = %73,5),
  train %59,93, **val %57,09**; 30. epoch'ta durdu.
- **Arena 1.000:** **%73,5** (735-265), 0 hata — **yeni Dragapult rekoru**.
  Mini-arena tahmini gerçek sonuçla birebir örtüştü.
- Karşılaştırma: 104k/patience2 %68,2 → 385k/patience2 %51,9 →
  **385k/patience5 %73,5**.
- `model.npz` = v12.3 (560,1 KB), paket yeniden üretildi; Banette ağırlıkları
  `model_v11_banette.npz`.

### v12.4 — Anti-meta ağırlıklandırma (2.0) — regresyon

- `build_dragapult_dataset.py` satır bazında `source` üretir: Dragapult
  galibiyetleri 0 (ağırlık 1.0), Dragapult'u yenen rakiplerin hamleleri 2
  (**ağırlık 2.0**) → 286.550 / 98.885 satır (etkin sinyalin %20,4'ü).
- Eğitim (v12.3 altyapısı): best epoch 20 (mini-arena %68,0), val %56,77.
- **Arena 1.000:** **%67,4** (674-326) → v12.3'ün %73,5'inin **6,1 pp**
  altında (≈3σ). Anti-meta satırları başka destelerin hamleleri olduğu için
  Dragapult politikasını seyreltiyor.
- `model.npz` = v12.3 korundu; v12.4 ağırlıkları
  `model_dragapult_v124.npz`.

### v12.4.1 — Anti-meta 0.5 ile fine-tune

- Aynı 385.435 satır; anti-meta 98.885 satır **0.5** ağırlıkla.
- **v12.3 ağırlıklarından** başlatıldı (`--init`, 81 tensör), lr **1e-4**,
  **5 epoch** → train %63,38, **val %57,82**.
- **Arena 1.000:** **%70,1** (701-299) — v12.4'ün (%67,4) üzerinde,
  v12.3'ün (%73,5) altında (−3,4 pp, ~1,6σ).
- Sonuç: düşük doz + fine-tune kaybın bir kısmını geri kazandırdı ama
  baseline'ı geçemedi. Dragapult–Banette eşleşmesinde anti-meta enjeksiyonu
  net negatif; `model.npz` v12.3'te bırakıldı.

### v12.5 — Sıfırdan, saf Dragapult galibiyetleri

- Veri: `dataset_dragapult_wins.npz` — yalnızca Dragapult'un kazandığı
  maçlar, **286.550 satır / 1.141 episode** (anti-meta 0, mağlubiyet 0).
- Eğitim: **rastgele init**, v12.3 altyapısı, lr 1e-3, patience 5,
  arena 200, 30 epoch → best **epoch 30** (mini-arena %65,5), val %53,88
  (kendi val split'inde), train %62,29.
- **Arena 1.000:** **%66,8** (668-332), 0 hata.
- Kıyas: v12.3 (%73,5, 385k eşit ağırlık) > v12.4.1 (%70,1) >
  v12.4 (%67,4) > **v12.5 saf-galibiyet (%66,8)**.
- Çıkarım: Dragapult'un kaybettiği maçlardaki rakip satırları (98.885)
  **eşit ağırlıkla** değerli (+6,7 pp); hem 2.0 hem 0.5 ağırlık hem de
  tamamen çıkarmak performansı düşürüyor. En iyi reçete v12.3'ün eşit
  ağırlıklı karışımı.

### v12.6 — "Sınırları Kaldır" (epochs 100, patience 10)

- Veri: v12.3 ile **aynı** — 385.435 satır / 1.788 episode, anti-meta
  ağırlığı **1.0** (eşit). Rastgele init, v12.3 mimarisi, arena 200.
- Eğitim tavanı 100 epoch, patience **10 değerlendirme** (= 50 epoch).
- Zirve: **epoch 35** — mini-arena %76,5 (153/200), val %56,91,
  train %63,18. 85. epoch'ta erken durdu.
- **Arena 1.000:** **%69,9** (699-301), 0 hata.
- Doygunluk eğrisi: arena 20→35 arası %76 bandında, 40'tan sonra
  %57–73 arası dalgalı; val loss 15. epoch'taki 1,1004 dip noktasından
  sonra tek yönlü yükseldi (85. epoch 1,3135) — net **overfit**.
- Kıyas: v12.3 **%73,5** > **v12.6 %69,9** > v12.4.1 %70,1 ≈ eşit.
  Fark ~1,75σ (n=1000'de SE ≈ 2,1 pp); yani "v12.3 daha iyi" güçlü
  ama kesin değil — asıl kanıt v12.6'nın val eğrisinin 15. epoch'tan
  sonra bozulması.
- Ders: tavan **eğitim süresi değil, veri + mimari**. 30 epoch zaten
  doygunluğa yakındı; 70 epoch daha görmek sadece train top-1'i
  (%63 → %69,8) şişirdi, arena'yı düşürdü. Mini-arena'nın 200 maçlık
  gürültüsü (SE ≈ 3,2 pp) + "en iyi değerlendirmeyi seçme" yanlılığı
  (%76,5 → gerçek %69,9) kayda değer bir seçim şansı yaratıyor.

### v12.7 — "Büyük Veri" (3.900 episode, 871k satır)

- Veri: Dragapult havuzu **1.800 → 3.900 episode** (leaderboard'un
  tamamı, 563 takım × 8 submission). Derleme: **871.573 satır /
  3.887 episode** (2.592 Dragapult galibiyeti + 1.308 mağlubiyetten
  rakip kararları), train 697.502 / val 174.071.
- Rakip dağılımı: Alakazam 330, Banette 319, Mega Excadrill 300,
  Hydrapple 170, Festival Lead 139 — yani havuz artık Banette merkezli
  değil, tüm meta.
- Eğitim: sıfırdan, 40 epoch tavanı, patience 5, arena 200, eşit ağırlık.
  Best **epoch 10** (mini-arena %60,0, val %56,27), 35. epoch'ta durdu.
  Sonrası: %41 → %52 → %50 → %49,5 → %54,5 (dalgalı, zirveyi geçemedi).
- **Arena 1.000:** **%60,4** (604-396), 0 hata.
- Kıyas: v12.3 **%73,5** >> v12.7 %60,4 (−13,1 pp). Mini-arena (%60,0)
  ile final (%60,4) bu kez **tutarlı** — yani düşüş seçim yanlılığı değil,
  modelin gerçek seviyesi.
- Çıkarım: **hacim tek başına işe yaramıyor.** Aynı 299k parametreli ağ,
  2,26× veriyi ve 30+ farklı eşleşmeyi taşıyamıyor; Banette'e özgü keskinlik
  seyreltiliyor. v12.3'ün 385k satırı Banette–Dragapult odaklıydı ve bu
  odak, hedef maçta daha değerliydi. Sıradaki kaldıraç: kapasite artışı
  veya büyük havuzda ön-eğitim + Banette alt kümesinde fine-tune.

### v12.8 — "Devasa Mimari" + Kaggle GPU hattı

- Mimari: **6 blok × 8 baş, d=128**, Pre-LN, FFN 128→400→128 →
  **1.140.065 parametre** (v12.3'ün 3,8 katı). CPU'da ~15 dk/epoch
  ölçüldü; eğitim Kaggle GPU'suna taşındı.
- `train_bc.py`: `pick_device()` + `--device auto|cpu|cuda`; model
  `.to(device)`, veri satırları host RAM'de kalıp **batch bazında** GPU'ya
  taşınıyor (871k × 41 × 41 option tensörü VRAM'e sığmak zorunda değil).
  `evaluate()` chunk'ları da cihaza taşır. CPU smoke testi geçti.
- Mini-arena checkpoint'i artık `--out` klasörüne yazılıyor
  (`_arena_candidate.npz`): `/kaggle/input` salt-okunur olduğu için zorunlu.
- Boyut: 1,14M parametre FP16'da **2.030 KB** — 600 KB kuralıyla
  çelişiyor. Çözüm: `src/mazlum/quantize.py` + `tools/mazlum/quantize_model.py`
  int4 paketleme (simetrik, per-tensor scale, iki değer/bayt) →
  **~589 KB**; `agent.py` arşivi açıp float32'ye çeviriyor ve mimariyi
  `arch` metadata'sından okuyor. NumPy↔PyTorch farkı: FP16 2e-4,
  int4 ~4e-2 (arena ile ölçülecek).
- Kaggle: veri + kod **private dataset** olarak yüklendi
  (`farhandir/mazlum-dragapult-v127`, 33,8 MB, 25 dosya: 871k npz,
  `code/src/mazlum`, `vendor/cg` Linux `.so` dahil, botlar).
  `train_kaggle.ipynb` veriyi `/kaggle/input/...`ten okur, GPU'da eğitir,
  int4 export alır ve 1.000 maçlık arenayı çalıştırır.
- v3 güncellemesi: `train_bc.py` varsayılan **batch 2048** (`--batch`,
  `--batch-size` alias) → 697k satırda ~341 adım/epoch; notebook
  **LR 3e-3**. Gerekçe: batch 32× büyürken adım sayısı 32× azaldığı için
  eski 1e-3 fiilen çok küçük kalıyordu (CPU koşusu 25 epoch'ta %49'da
  takıldı). Notebook'a erken uyarı eklendi: epoch 5 arena'sı %50 altında
  kalırsa LR'yi 1e-3'e düşür.

### v12.9 — "Eski Usul": v12.3 + Banette-only fine-tune

- v12.8/Kaggle hattı iptal; mimari **4 blok × 4 baş, d=80, 299.329
  parametre**, batch **64**'e döndü (CUDA plumbing `--device auto` olarak
  kaldı, CPU'da etkisiz).
- Banette verisi: `--opponent Banette` filtresiyle **145.720 satır /
  725 episode** (319 Dragapult galibiyeti, 406 mağlubiyet). Anti-meta
  payı **%38,5** — v12.3'ün karışımında %25,7.
- v12.9: lr 1e-4, 12 epoch, sıfırdan değil `model.npz`'den fine-tune.
  **Arena: %70,2** (702-298). Kayıp nedeni: karışım kayması + tek arena
  değerlendirmesi (aşağıdaki kadans hatası).
- **Kadans hatası (düzeltildi):** arena `epoch % arena_every` ile
  tetikleniyordu ama metrik bloğu `print_every`'ye bağlıydı; 3 ve 5'in
  LCM'i 15 olduğu için 12 epoch'luk koşuda arena **yalnızca 1 kez** çalıştı,
  `patience` hiç devreye girmedi. Artık arena kendi kadansında çalışıyor.
- v12.9b (düzeltilmiş kadans, aynı veri): arena 3/6/9/12 →
  %64,5 / **%70,0** / %68,5 / %66,5; best **epoch 6**.
  **Final arena: %72,9** (729-271), 0 hata — şampiyona **0,6 pp** kala.
  Mini-arena (%70,0) finali yine ~3 pp düşük tahmin etti (200 maç gürültüsü).
- Çıkarım: Banette-only fine-tune v12.3'ün seviyesini **koruyor** ama
  geçmiyor; 145k satırlık dar set, 385k karışık setin yerini tutmuyor.
  Şampiyon `model.npz` (v12.3, %73,5) değişmedi.
- v12.9c (anti-meta **0.5**, v12.3'ün 2,9:1 oranına dönüş): arena
  3/6/9/12 → %69,0 / %75,5 / **%76,5** / %67,5; best epoch 9.
  **Final arena: %71,4** (714-286). Mini-arena yine +5 pp şişirdi.
- Üç fine-tune'ın özeti: 70,2 / 72,9 / 71,4 — hiçbiri %73,5'i geçmedi.
  Ölçümlerin tümü aynı bantta (70–73); v12.3'ün 73,5'i bu bandın üst
  ucunda. Mimari/veri seti değişmeden tavana yaklaşıldı.
- **v12.3 sağlama (yeni seri):** `arena.py`'ye `--shuffle` eklendi (ilk
  oyuncu `--seed`'den yazı-tura). Motorun Python API'sinde seed yok;
  iki ardışık 20 maçlık seri birebir aynı çıkmadı → her süreç zaten
  taze dağıtım yapıyor.
  Yeni koşu: seed **20261029**, 1.000 maç → **%75,8** (758-242), 0 hata.
  Yani %73,5 **şanslı bir çekiliş değil**, aksine alt sınır. İki seri
  birleşik: **1.493/2.000 = %74,7**. 73,5 ile 75,8 arasındaki fark
  anlamsız (z ≈ 1,2); gerçek seviye ~%73-78 aralığında.

### v13 — Özellik mühendisliği (37 boyut)

- Banette hattı v9 kod tabanına (2 blok × 2 baş, 147.521 parametre)
  döndü; şampiyon **v11 %87,4** korundu.
- **Yeni state vektörü: 73 → 37.** 64'lük deste parmak izi 16'ya indirildi
  (688k satırda yalnızca 162 benzersiz histogram vardı), boşalan 48 boyut
  yerine 12 dinamik stratejik sinyal eklendi:
  `opp_hand_count` (Poltergeist), `my/opp_prizes`, `my_prize_progress`,
  `opp_prize_threat`, `my/opp_deck_ratio`, `my/opp_discard_supporters`,
  `my/opp_discard_boss` (Boss's Orders=1182), `opp_bench_energy`.
  Ayrıca `_hand_count` hatası düzeltildi (rakip `hand` listesi boş,
  `handCount` doğru kaynak).
- Veri: aynı 1.800 episode (hardlink) → **680.104 satır / 37 boyut**
  (`dataset_v13.npz`, 27,9 MB). `source` kolonu yok → ağırlık 1.0.
- Eğitim: sıfırdan, lr 1e-3, batch 64, ≤80 epoch, patience 2,
  arena 5 × 200. Best **epoch 15** (mini-arena **%95,5**), train %62,38,
  val %57,73; epoch 25'te durdu. Model **532,7 KB**.
- **Arena 1.000: %94,8 (948-52), 0 hata.** v11'e karşı **+7,4 pp**
  (z ≈ 5,9 → kesin). Val top-1 neredeyse sabit (%56,74 → %57,73):
  kazanç taklitten değil, **karar kalitesinden**.
- Artefaktlar: `data/mazlum/model_banette_v13.npz`,
  `data/mazlum/dataset_v13.npz`,
  `results/mazlum_v13_banette_single_1000.json`,
  `tools/mazlum/patches/v13_features_37dim.patch`.
- **Yarışma paketi:** worktree'de `tools/build_submission.py` ile
  `submissions/mazlum_v13_banette.tar.gz` → **2,56 MiB** (sınır 197,7 MiB).
  İçerik doğrulandı: `model.npz` 532,7 KB / `state_proj (64,37)`,
  `features.py` 37-dim, Banette destesi, `cg/`. İzole dizine açılıp
  oynatıldı: vs Banette botu 18/20, 0 hata.
- **Çapraz ateş (Dragapult heuristik botu, 1.000 maç, seed 777):**
  v13 **%42,4** (424-576) / v11 **%31,5** (315-685) → **+10,9 pp**
  (z ≈ 5,1). Yani özellik mühendisliği yalnızca ayna eşleşmesini değil,
  meta eşleşmesini de iyileştirdi. Mutlak değerin %50 altında olması
  deste eşleşmesinden kaynaklanıyor (Banette destesi Dragapult'a karşı
  yapısal olarak zor); modelin genel oyun kalitesi yine de v11'in
  belirgin üstünde.

### v14 — Çok-deste eğitimi (900k satır, 37-dim)

- Veri: manifestin karıştırılmış sırasından **2.197 episode / 900.000
  satır / 40 archetype**, her iki tarafın kararları, 37-dim.
  Yeni `tools/build_multi_dataset.py` çıktı dizilerini önceden ayırıp
  dolduruyor (liste+stack yaklaşımı 900k satırda ~6 GB isterdi).
- Eğitim: sıfırdan, lr 1e-3, batch 64, ≤80 epoch, patience 2,
  arena 5 × 200. Best **epoch 5** (mini %95,0), epoch 15'te durdu.
  Train %57,0 / val %55,5 → **doygunluğa ulaşmadan kesildi** (~56k adım;
  v13 ~128k adım görmüştü).
- **Bot bataryası (500 maç, seed 1234, aynı koşullar):**

  | Model | Banette | Dragapult | Lucario |
  |---|---:|---:|---:|
  | v13 (Banette-only) | %94,6 | **%45,2** | **%74,8** |
  | v14 (multi-deck) | %94,6 | %33,6 | %70,6 |

- Sonuç: bu koşuda çok-deste eğitimi **kazanç getirmedi** (Dragapult
  −11,6 pp, z ≈ 3,8; Lucario −4,2 pp, anlamsız). Ancak deney **kusurlu**:
  (a) erken durdurma epoch 5'te takıldı, (b) seçim metriği yalnızca
  Banette botu, (c) v9 kodunda eşitlikte ilk model korunuyor (v10'un
  `>=` + val tie-break yaması yok). Temiz bir tekrar için bu üçü
  düzeltilmeli.

### v14b — Düzeltilmiş çok-deste koşusu

- Yamalar: `>=` + val tie-break (val top-1, eşitse val loss),
  yeni `--arena-start` (arena 10. epoch'tan itibaren), arena 400 maç,
  patience 4. Metrik/arena bağımlılığı da giderildi.
- Eğitim: best **epoch 10** (mini %95,5 / 400), epoch 30'da durdu.
  Arena eğrisi: 95,5 → 94,2 → 92,0 → 91,5 → 93,0; val top-1 ise
  **epoch 25'te 57,13** ile zirve yapıyor. Yani Banette-arenası
  çok-deste modelde **yanlış sinyal**: erken epoch'ları ödüllendiriyor.
- **Batarya (500 maç, seed 1234):**

  | Model | Banette | Dragapult | Lucario |
  |---|---:|---:|---:|
  | **v13** | %94,6 | **%45,2** | **%74,8** |
  | v14 | %94,6 | %33,6 | %70,6 |
  | v14b | %94,6 | %37,6 | %71,8 |

- Sonuç: düzeltmeler v14'ü iyileştirdi (+4,0 / +1,2 pp) ama **v13'ü
  geçemedi** (−7,6 / −3,0 pp). Çok-deste hipotezi bu mimaride
  (147k parametre) **doğrulanmadı**: model deste-koşullu ve kapasite
  desteler arasında bölünüyor. Üretim hattı v13'te kalıyor.

### v15 — Hedefli madencilik (Banette vs Dragapult) + fine-tune

- **Madencilik:** `tools/mazlum/mine_targeted.py` yalnızca Banette ∩
  Dragapult episode'larını indirir. 47 dk / 563 takım taraması →
  **+81 yeni episode** (725 → **806**).
- **Veri:** `dataset_v15_dragapult_focus.npz` — 806 episode,
  **322.160 satır** (256.282 train / 65.878 val), 37-dim, her iki tarafın
  kararları. 12,4 MB.
- **Fine-tune:** v13 ağırlıklarından, lr **1e-4**, 10 epoch, arena kapalı
  (Banette-arenası bu iş için yanlış sinyal), her epoch checkpoint.
  Val top-1 %57,72 → **%58,61**; train %57,64 → %60,45.
- **Checkpoint taraması** (Dragapult botu, 200 maç):
  ep1 %50,0 · ep2 %53,5 · ep3 %56,5 · ep4 %56,5 · ep5 %47,5 · ep6 %52,5
  · **ep7 %60,5** · ep8 %55,5 · **ep9 %59,0** · ep10 %54,5.
  1.000 maç doğrulaması: ep7 **%51,1**, **ep9 %55,6** (v13: %42,4,
  aynı seed 777) → **+13,2 pp** (z ≈ 6,0).
- **Final batarya (500 maç, seed 1234):**

  | Model | Banette | Dragapult | Lucario |
  |---|---:|---:|---:|
  | v13 | %94,6 | %45,2 | %74,8 |
  | **v15 (ep9)** | **%96,2** | **%56,0** | **%77,8** |

  Üç eşleşmede de iyileşme: **+1,6 / +10,8 / +3,0 pp**. Hedefli fine-tune
  ayna eşleşmesini bozmadı, güçlendirdi.
- Paket: `submissions/mazlum_v15_banette.tar.gz` — **2,56 MiB**, içindeki
  `model.npz` epoch_009 ile birebir (SHA eşleşti), izole dizinde 20/20.

### v16 — Kaggle meta madenciliği (özeleştiri fine-tune)

- Kaynak: kendi submission'larımız (57835594 Banette dönemi + 57643520
  Dragapult dönemi) → **637 yeni episode** indirildi (+246 / +391).
- Veri: `dataset_v16_kaggle_meta.npz` — 637 episode, **198.661 satır**
  (154.808 train / 43.853 val), 37-dim. Val top-1 %63,90 → %65,56 (ep5).
- Fine-tune: v15 ağırlıklarından, lr 1e-4, 10 epoch, her epoch checkpoint.
- Checkpoint taraması (Dragapult, 200 maç): en iyi ep2/ep8 **%50,5**;
  ep3 %40,0, ep7 %38,5. v15'in %56,0'ının altında.
- Final batarya (ep8, 500 maç): **Banette %91,0 · Dragapult %47,8 ·
  Lucario %75,0** — üçü de v15'in altında (−5,2 / −8,2 / −2,8 pp).
- Sonuç: **v16 reddedildi.** Kendi Kaggle maçlarımızdan öğrenmek, modeli
  kendi dağılımına çekip hedefli uzmanlıkları (Dragapult savunması, ayna)
  seyreltti. İlk epoch'ta bile Dragapult %46'ya düştü → dağılım kayması
  hızlı ve geri dönüşsüz. Şampiyon **v15** kalıyor.

### v17 — Hedefli veriyi dikey ölçekleme (1.705 episode)

- Madencilik: yalnızca Banette oynayan submission'lar, 300 episode/sub,
  420 dk bütçe. **881 → 1.705** Banette-vs-Dragapult episode (+824).
  Kaggle 429 kotası nedeniyle 2.000'e ulaşılamadı (9,4 saat, 520 adet 429).
- Veri: `dataset_v17_dragapult_massive.npz` — **671.486 satır**
  (539.258 train / 132.228 val), 37-dim, 25,8 MB.
- Eğitim: v13 ağırlıklarından, lr 1e-4, 15 epoch (12 + kaldığı yerden 3),
  checkpoint'li. Val top-1 %58,27 → **%59,52** (epoch 12'den sonra plato).
- Tarama (15 checkpoint × 200 maç Dragapult): zirve **epoch_006 %60,0**;
  6 checkpoint %57,5-60 bandında (v15'te 2 taneydi).
- Doğrulama ve batarya:
  - 1.000 maç (seed 777): **%57,7** (v15: %55,6)
  - Batarya (500, seed 1234): Banette **%96,2** · Dragapult **%52,4** ·
    Lucario **%75,8**
  - Birleşik Dragapult (1.500 maç): v15 **%55,7** vs v17 **%55,9** →
    **fark yok** (0,2 pp).
- Sonuç: hedefli veriyi 806 → 1.705 episode'a çıkarmak **ölçülebilir
  kazanç getirmedi**. Veri ölçeği bu mimaride doygunluğa ulaştı; ayna
  korunuyor (%96,2), Lucario hafif düşük (farksız). Şampiyon **v15**.

### v18 — Akıllı Kaggle meta (yalnızca kazanılan maçlar)

- Çekim: son submission (57957138, v15) → 132 episode, 127 indirildi;
  bunlardan 60'ı galibiyet (%47,2 — yüksek Elo duvarı).
- Filtre: **yalnızca Banette'in kazandığı** maçlar + leaderboard ilk 200
  takımın Banette zaferleri. 1.174 mağlubiyet dışlandı.
- Seçilen: **1.894 episode** (60 bizim + 1.834 üst takımlar). Rakipler:
  Dragapult 910, Alakazam 268, Festival Lead 186, Banette 150, Hydrapple 148.
- Veri: `dataset_v18_kaggle_smart.npz` — **732.194 satır**
  (590.075 train / 142.119 val), 37-dim.
- Eğitim: v15 ağırlıklarından, lr 1e-4, 10 epoch. Val top-1
  %59,74 → **%60,04** (ep5) → %59,93.
- Tarama (10 checkpoint × 200): en iyi **epoch_007 %56,5**.
- Batarya (500): **Banette %96,0 · Dragapult %50,6 · Lucario %78,2**
  (v15: %96,2 · %56,0 · %77,8).
- Sonuç: ayna korundu (+0,0), Lucario sınırda iyileşti (+0,4, anlamsız),
  ama **Dragapult −5,4 pp**. Kazanılan-maç filtresi de v16/v17 desenini
  değiştirmedi: kendi/Kaggle dağılımından fine-tune, yerel heuristiğe
  karşı öğrenilmiş Dragapult uzmanlığını seyreltiyor. Şampiyon **v15**.

### v19 — "Büyük Beyin" (4×4×80 Pre-LN, 37-dim, cold start)

- Mimari: v12.3'ün Pre-LN gövdesi **37-dim** state ile birleştirildi
  (`MazlumNetBig`): 4 blok × 4 baş, d=80, FFN 144, head 160→128→1.
  **296.449 parametre** (v15'in 2,0 katı).
- Boyut: FP16 **549-553 KB** (< 600 KB), FP32 1.071 KB. NumPy↔Torch
  farkı 8,9e-8; ajan iki mimariyi de anahtarlardan otomatik tanıyor.
- Eğitim: `dataset_v17` (671.486 satır), **sıfırdan**, lr 3e-4, 15 epoch,
  her epoch fp16 checkpoint. Val top-1 %56,43 → **%59,86** (ep12) →
  %59,58; train %64,05.
- Tarama (15 × 200 Dragapult): zirve **ep14 %64,0** (v15: %60,5).
- Batarya (ep14, 500): **Banette %94,0 · Dragapult %59,8 · Lucario %74,0**.
- 1.000 maç doğrulama (seed 777): Dragapult **%56,3**, ayna **%93,3**.
- Havuzlanmış (1.500 maç): Dragapult **v19 %57,5 vs v15 %55,7**
  (+1,8 pp, 1,0σ → anlamsız); ayna **%93,5 vs %95,3** (−1,8 pp, ~2,1σ).
- Sonuç: kapasite artışı Dragapult'ta **yön olarak olumlu ama anlamsız**,
  aynada ise **ölçülebilir küçük kayıp**. Karar kuralı (Dragapult ≥%58 ve
  ayna ≥%94) sağlanmadı → **v15 şampiyon**. v19 arşivde.

### v20 — Büyük beyin + hedefli veri (kapasite ∩ uzmanlık)

- v19 mimarisi (296.449 param, 4×4×80 Pre-LN, 37-dim), `dataset_v15`
  (322.160 satır / 806 episode) üzerinde **v19'dan fine-tune**:
  lr 1e-4, 10 epoch, fp16 checkpoint.
- Val top-1: **%62,71** (ep1) → %61,15 (ep10) — hedefli sette ezber
  başlıyor; arena yine farklı yerde zirve yaptı.
- Tarama (10 × 200 Dragapult): zirve **ep7 %58,0**.
- **Altın doğrulama (1.000 maç, seed 777):**
  Dragapult **%58,6** (586-414) · ayna **%92,3** (923-77).
- Karar kuralı: Dragapult ≥%58 ✔ / ayna ≥%94 ✘ → **v20 reddedildi**.
  Dragapult farkı +3,0 pp (1,4σ, anlamsız); ayna −2,5 pp (2,3σ, anlamlı).
- Kapanış: kapasite artışı Dragapult'u hafifçe yukarı, aynayı **gerçekten**
  aşağı çekiyor. İki eksen birlikte kazanılamıyorsa turnuva için doğru
  tercih ayna ağırlıklı **v15**.

### v21 — Kayıp eşleşmelere hedefli fine-tune (Hydrapple/Archaludon/Festival/Grimmsnarl/Lopunny)

- Veri: 498 episode → **174.771 satır** (139.399 train / 35.372 val).
  Dağılım: Festival 219, Hydrapple 204, Archaludon 45 (yalnız 5 Banette
  galibiyeti), Grimmsnarl 19, Lopunny 11.
- v15'ten fine-tune: lr 1e-4, 8 epoch, fp32. Val top-1 %65,66 (ep4'te plato).
- Tarama (8 × 200 Dragapult): zirve ep2 %52,5.
- **Altın doğrulama (1.000 maç, seed 777):**
  Dragapult **%47,4** (−8,2 pp) · ayna **%93,2** (−1,6 pp).
  Yerel protokol "reddet" dedi; blog dersi gereği yine de ladder'a gönderildi.
- **Ladder sonucu:** takım skoru **896,2 → 939,0** (sıra 196 → **148**), +42,8.
  Yerel Dragapult regresyonu ladder'da **tersine döndü**. v21 ikinci slotta kaldı.
- Ders: yerel metrik iki kez üst üste yanıltıcı; hedefli veri ladder'da çalışıyor.

### v22 — Slowking BC uzmanı (sıfırdan, 37-dim)

- Veri: 513 Slowking-Kangaskhan episode → **168.720 satır**
  (136.738 train / 31.982 val). Deste: `decks/lab/slowking_kangaskhan.csv`
  (60 kart, Akademi + Ciphermaniac + Kyurem/Conkeldurr/Annihilape + Mega
  Kangaskhan + Lillie's Clefairy).
- Sıfırdan `MazlumNet` (147.521 param), lr 1e-3, 25 epoch, her epoch checkpoint.
  En iyi val loss **ep23** (top-1 %58,85); train %65,06.
- Yerel (100 maç, seed 4242): Banette'e **%55,0** (4 oyun `max_steps` draw),
  Dragapult'a **%20,0** — yerel bot Dragapult'a karşı zayıf, ama bu ölçüt
  ladder'ı öngörmüyor.
- Paket: `submissions/mazlum_v22_slowking.tar.gz` (2,56 MiB, model sha
  `871246e5`, deste 60/60 doğrulandı).
- Kaggle **ref 58031843** (A/B; Alakazam "known 1074.1" birinci slotta).
- Karar: 24 saat sonra takım skoru > 939 ise Slowking hattı devam;
  düşerse v21 paketi geri yüklenir.

### Kaggle gerçeği (5 Kasım 2026)

- Takım: **"Attention Is All You Mew"**, iki skorlu slot = **en son iki submission**.
- Aktif çift: Alakazam PPO (ref 58030010, bilinen 1074,1) + son deneme.
- Ölçülen: 896,2 (sıra 196) → v21 ile **939,0 (sıra 148)**.
- Lider: Petit Canard 1574,3. Aradaki fark ~635 puan; BC-only ile kapanmaz,
  lig PPO + uzmanlaşma gerekir.

















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
