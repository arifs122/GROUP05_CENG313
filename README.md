# TransforMew

Pokémon TCG AI Battle Challenge için geliştirilen oyun ajanı.

## Özet

Kaggle [PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
ortamında çalışan bir ajan. Simülatörden gelen gözlemi alır, yasal seçenekler
arasından karar verir ve seçenek indekslerini döndürür. Maçlar BO3 formatında
oynanır; değerlendirme galibiyet/mağlubiyet üzerinden yapılır.

Repo şunları içerir:

- yerel simülasyon ortamı (`vendor/cg`, kurulumla oluşur),
- baseline politika (`src/agent`),
- maç ve ölçüm araçları (`tools/`),
- Kaggle submission paketleyicisi.

## Kurulum

Python 3.10+ gerekir.

```bash
git clone https://github.com/arifs122/TransforMew.git
cd TransforMew
bash scripts/setup_assets.sh /path/to/sample_submission/cg
python3 scripts/check_env.py
```

Yarışma motoru ve kart verisi lisans nedeniyle repoda tutulmaz. Script, verilen
`cg/` klasörünü `vendor/cg` altına, kart CSV'lerini `data/` altına kopyalar.

## Kullanım

```bash
# yerel maç
python3 tools/arena.py --games 100

# gözlem inceleme
python3 tools/inspect_obs.py --steps 20

# submission paketi
python3 tools/build_submission.py
```

Submission çıktısı `submissions/submission.tar.gz` olur; kökte `main.py`,
`deck.csv` ve `cg/` bulunur.

## Botlar

Her bot `bots/<bot_adi>/` altında kendi `main.py` ve `deck.csv` dosyasıyla
durur. Şablon:

```bash
cp -r bots/_template bots/benim_botum
# main.py, deck.csv ve meta.json düzenle
```

```bash
# iki botu karşılaştır
python3 tools/arena.py --bot0 bots/benim_botum --bot1 bots/_template --games 50

# BO3
python3 tools/arena.py --bot0 bots/benim_botum --bot1 bots/_template --games 30 --bo3

# tüm botlarla round-robin turnuva
python3 tools/tournament.py --games 50

# seçilen botu paketle
python3 tools/build_submission.py --bot bots/benim_botum
```

Ayrıntılar ve kurallar: [`bots/README.md`](bots/README.md)

## Yapı

```
src/agent/       varsayılan ajan ve politika
bots/            ekip botları (her biri main.py + deck.csv)
decks/           ortak deste dosyaları
tools/           arena, turnuva, gözlem inceleme, paketleme
scripts/         kurulum ve kontrol
tests/           smoke test
```

## Lisans

Bu repodaki özgün kod MIT lisansı ile dağıtılır (bkz. `LICENSE`).
Pokémon kartları, `cabt` motoru ve yarışma verisi bu lisansın kapsamı dışındadır;
Kaggle yarışma kurallarına tabidir ve repoda yer almaz.
