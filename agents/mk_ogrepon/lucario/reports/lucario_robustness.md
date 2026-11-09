# Lucario — legalite / crash / süre

## Rakip havuzu

`tools/eval_pool.py --games 20` (7 rakip, 140 maç):

- **0 geçersiz aksiyon**
- **0 exception**
- **0 engine error**
- Tüm maçlar tamamlandı.

## BO3

- `bots/_template`, 50 BO3: **27-23 (%54)**, 0 error
- `bots/dragapult`, 20 BO3: **0-20 (%0)**, 0 error
- Ortalama BO3 karar adımı: 174 / 292

## Tek maç

- Kullanıcı destesi vs template 100 maç: seed'e göre %34-51
- `deck_riolu974`: %40-46
- `deck_pokepad`: %39-47
- `deck_real_petit`: %34-47

Bu sayılar lokal simülasyon içindir; ladder kabul kriteri değildir.

## Adaptör güvenliği

`agent()` her durumda:

1. `select is None` → desteyi döndürür.
2. MAIN/single seçimde `score` + `choose`.
3. Çoklu seçimde `_pick_multi` (discard'te en düşük değerli kart).
4. Exception halinde `_fallback` ile legal ilk seçenek.

`to_acts` motorun `Option` dict/dataclass yapısını `mk_state` üzerinden okur;
gizli rakip eli/prize/deck bilgisi kullanılmaz.
