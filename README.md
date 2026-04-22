<<<<<<< HEAD
# 💊 RxGuard — Medical Prescription Error Detection System

**Fine-tuned DistilBERT NLP classifier · Runs 100% locally on CPU · No cloud APIs required**

---

## 📁 Project Structure

```
med_prescription_error/
│
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator (≈480 rows)
│   └── prescriptions.csv        # Generated after running the script
│
├── model/                       # Created after training
│   ├── config.json
│   ├── pytorch_model.bin (or model.safetensors)
│   ├── tokenizer files …
│   └── training_meta.json
│
├── backend/
│   └── app.py                   # Flask REST API  (POST /predict)
│
├── frontend/
│   └── index.html               # Single-page UI (no framework needed)
│
├── utils/
│   ├── __init__.py
│   └── predictor.py             # Inference wrapper
│
├── train.py                     # Fine-tuning script
├── test_api.py                  # CLI test suite
└── requirements.txt
```

---

## 🧠 Problem Statement

Given free-text prescription notes, classify them as:

| Label | Description |
|---|---|
| `safe` | Normal dosage, no interactions detected |
| `overdose` | Dose significantly exceeds safe thresholds |
| `drug_interaction` | Two or more drugs with known dangerous interaction |

---

## ⚙️ Setup Instructions

### 1. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱ First run downloads the DistilBERT weights (~260 MB) from HuggingFace.

---

### 3. Generate Dataset

```bash
python data/generate_dataset.py
```

Output: `data/prescriptions.csv` with ~480 labelled rows (balanced 3-class).

---

### 4. Train the Model

```bash
python train.py
```

Optional flags:
```bash
python train.py --epochs 6          # more training
python train.py --data data/custom.csv
```

Expected output:
```
Epoch 1/4  |  Train Loss: 0.8432  Acc: 0.6823  |  Val Loss: 0.5214  Acc: 0.8125  |  ⏱ 45.2s
Epoch 2/4  |  Train Loss: 0.4112  Acc: 0.8579  |  Val Loss: 0.3041  Acc: 0.9167  |  ⏱ 44.8s
...
🎉  Training complete!  Best val accuracy: 0.9479
   Model saved to → model/
```

> ⏱ Each epoch takes ~40-90 seconds on a modern CPU.

---

### 5. Run the Backend

```bash
cd backend
python app.py
```

You should see:
```
INFO  Loading model from .../model …
INFO  Model loaded ✓
INFO  Starting Flask server on http://localhost:5000
```

---

### 6. Open the Frontend

Just open `frontend/index.html` in your browser — no server needed.

Or serve it:
```bash
# Python built-in server (from project root)
python -m http.server 8080
# Then visit: http://localhost:8080/frontend/
```

---

## 🧪 Testing

### Run the test suite (backend must be running):

```bash
python test_api.py
```

### Test a custom prescription:
```bash
python test_api.py --text "Patient prescribed Codeine 300mg every 4 hours for pain."
```

### Manual cURL test:
```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Take Ibuprofen 400mg twice daily with food for mild arthritis pain."}' \
  | python -m json.tool
```

---

## 🧪 Sample Inputs & Expected Outputs

### ✅ Safe Prescription
```
Input: "Administer Lisinopril 10mg once daily for hypertension management."

Output:
{
  "label": "safe",
  "confidence": 0.9412,
  "probabilities": { "safe": 0.9412, "overdose": 0.0341, "drug_interaction": 0.0247 },
  "explanation": "This prescription appears within normal dosage guidelines …",
  "severity": "green"
}
```

### 🔴 Overdose
```
Input: "Patient prescribed Digoxin 1.5mg twice daily for atrial fibrillation."

Output:
{
  "label": "overdose",
  "confidence": 0.9183,
  "probabilities": { "safe": 0.0421, "overdose": 0.9183, "drug_interaction": 0.0396 },
  "explanation": "⚠️ The dosage specified exceeds the recommended safe threshold …",
  "severity": "red"
}
```

### 🟠 Drug Interaction
```
Input: "Continue Warfarin 5mg once daily and start Aspirin 75mg twice daily for cardiac protection."

Output:
{
  "label": "drug_interaction",
  "confidence": 0.9561,
  "probabilities": { "safe": 0.0211, "overdose": 0.0228, "drug_interaction": 0.9561 },
  "explanation": "⚠️ Two or more drugs in this prescription may interact dangerously …",
  "severity": "orange"
}
```

---

## 🚀 API Reference

### `GET /health`
Returns model status.

### `POST /predict`
**Request:**
```json
{ "text": "<prescription text>" }
```

**Response:**
```json
{
  "label":         "safe | overdose | drug_interaction",
  "confidence":    0.95,
  "probabilities": { "safe": 0.95, "overdose": 0.03, "drug_interaction": 0.02 },
  "explanation":   "Human-readable explanation string",
  "severity":      "green | red | orange",
  "input_text":    "original input text"
}
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: model/` | Run `python train.py` first |
| `Cannot reach backend` | Make sure Flask is running on port 5000 |
| Slow training | Normal on CPU; ~40-90s/epoch for ≈400 samples |
| Low accuracy | Try `--epochs 6` or increase dataset size in `generate_dataset.py` |
| Port 5000 in use | `export PORT=5001` then update `API_URL` in `frontend/index.html` |

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.  
Do **not** use in real clinical settings without extensive validation by qualified medical professionals.
=======
# rxguard-prescription-error-detection
💊 Medical Prescription Error Detection using Fine-Tuned DistilBERT | NLP | Flask | 96.88% Accuracy | CPU-only
>>>>>>> a76eed08aee88e625faae1ebc54f6af4603de15f
