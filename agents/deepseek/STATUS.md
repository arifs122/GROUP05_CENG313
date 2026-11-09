# DeepSeek workspace — durum

Tarih: 2026-11-05

## Mevcut durum
- Yeni submission gönderilmedi.
- `top2_core` submission ref: `57922537`, skor: `176.0` (COMPLETE).
- Karar: Top2 Dragapult'tan Grass/Ogerpon deck'ine geçiş.

## Grass/Ogerpon paketi (hazır)
- Deck: `decks/grass_ogerpon_kangaskhan.csv`
- Dataset: `data/dataset_grass_ogerpon.npz`
  - 332 episode, 822 game, 68.671 karar
  - 55.199 train / 13.472 val
  - state 78, option 68, max_options 40
- Notebook: `notebooks/train_grass_bc.ipynb` (Kaggle-ready, self-contained)
- Eğitilmiş modeller:
  - `models/value_model_grass.npz`
  - `models/policy_model_grass.npz`
- Notebook referans metrikleri:
  - value AUC ~0.740, BCE ~0.618
  - policy top-1 ~0.526, top-3 ~0.828
  - PyTorch -> NumPy parity geçti

## Sıradaki iş
1. Grass runtime bot: `profile='grass'` feature'ları + model çıkarımı.
2. Rule core'u Grass deck'e uyarla.
3. 200 BO3/rakip lokal test; submission yalnızca kullanıcı onayıyla.
