# Style Genie

Style Genie is a Flask fashion-recommendation experience that takes a portrait, detects a face, presents age/gender/skin-tone attributes, then curates an occasion-specific outfit.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Model integration

`utils/predict.py` checks image quality, detects faces with OpenCV, estimates skin tone from the face crop, and predicts age and gender with a ResNet50 trained on UTKFace. The app loads `models/style_genie_resnet50.pth` when present. The trained checkpoint is included. The raw UTKFace photos are not included in this public repository; to retrain, place a locally obtained UTKFace dataset in `UTKFace/` and run from the project directory:

```powershell
python -c "from utils.predict import MODEL_PATH, _train_and_save; _train_and_save(MODEL_PATH)"
```

Training uses a frozen ImageNet ResNet50 backbone and trains the age and gender heads on a randomized 4,000-image subset to keep CPU use practical. The checkpoint is saved in half precision to reduce file size; inference loads it into the normal full-precision model. Without a checkpoint, the app asks for age and gender instead of presenting heuristic guesses as model predictions.

`utils/recommend.py` loads the curated outfit rows from `datasets/fashion_dataset.csv` and matches age group, gender, and occasion. Skin tone is included in the recommendation result.

## Project structure

- `app.py` — Flask routes and image lifecycle
- `utils/predict.py` — image checks and face analysis
- `utils/recommend.py` — occasion data and recommendation service
- `templates/` and `static/` — responsive frontend
