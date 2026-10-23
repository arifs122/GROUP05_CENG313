# Botlar

Her bot kendi klasöründe, kendi kendine yeterli olacak şekilde durur.

## Klasör düzeni

```
bots/
  _template/          # başlangıç şablonu (turnuvaya girmez)
    main.py
    deck.csv
    meta.json
  benim_botum/
    main.py
    deck.csv
    meta.json         # opsiyonel
```

## Sözleşme

- `main.py` şu fonksiyonu dışa açmalı:

  ```python
  def agent(obs_dict: dict) -> list[int]:
      ...
  ```

- `agent`, `main.py` içinde **en son tanımlanan fonksiyon** olmalı. Kaggle
  doğrulama koşucusu modüldeki son callable'ı çağırıyor; `agent`'tan sonra
  `reset()` gibi bir fonksiyon tanımlarsanız doğrulama `agent` yerine onu
  çağırır ve submission `INVALID` olur (`Validation Episode failed`).
- `deck.csv` tam **60 kart** içermeli (her satırda bir kart ID).
- Bot klasörüne `cg/`, kart CSV'si veya PDF **konmaz**. Arena `vendor/cg`'yi
  kullanır.
- Oyun kuralları değiştirilmez. Bot sadece karar mantığını değiştirir.
- Bot mümkünse tek `main.py` olsun. Yardımcı modül kullanacaksanız
  `benim_botum_policy.py` gibi **benzersiz** bir ad verin ve importları dosya
  başında yapın.
- `meta.json` opsiyonel:

  ```json
  {
    "name": "benim_botum",
    "author": "github-kullanici-adin",
    "notes": "kısa strateji notu"
  }
  ```

## Yeni bot oluşturma

```bash
cp -r bots/_template bots/benim_botum
# main.py, deck.csv ve meta.json'u düzenle
```

## Çalıştırma

```bash
# iki botu karşılaştır (her biri kendi deck'iyle)
python3 tools/arena.py --bot0 bots/benim_botum --bot1 bots/_template --games 50

# resmî format BO3
python3 tools/arena.py --bot0 bots/benim_botum --bot1 bots/_template --games 30 --bo3

# tüm botlarla round-robin turnuva
python3 tools/tournament.py --games 50

# turnuva, BO3
python3 tools/tournament.py --games 30 --bo3

# seçilen botu Kaggle paketine çevir
python3 tools/build_submission.py --bot bots/benim_botum
```

Turnuva çıktıları `results/` altına yazılır (git'e girmez).

## İki lig

- **Açık lig (varsayılan):** Her bot kendi deck'iyle oynar. Sonuç
  "bot + deck paketi"nin gücüdür.
- **Sabit deck ligi:** `--deck decks/mega_abomasnow.csv` ile tüm maçlarda aynı
  deck kullanılır. Bu, sadece karar mantığını karşılaştırır. Not: bot kendi
  `deck.csv`'sini search/planlama için okuyorsa bu lig tam olarak izole
  olmayabilir; deney öncesi deck dosyalarını eşitleyin.

## Kurallar ve adalet

- Aynı motor (`vendor/cg`) ve aynı kart havuzu kullanılır.
- Yeterli maç oynatın: hızlı denemede 20–50, final sıralamada 100+.
- Taraflar dönüşümlü oynatılır; tek maçla karar vermeyin.
- Bir bot hata verirse turnuvada `error` olarak sayılır, sıralamaya sokulmaz.
- Rakip botların kodunu ancak PR ile ve sahibinin bilgisiyle değiştirin.
