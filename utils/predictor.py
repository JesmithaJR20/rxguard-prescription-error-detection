"""
utils/predictor.py  –  Inference wrapper around the fine-tuned DistilBERT model
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Plain-English explanations shown in the UI
EXPLANATIONS = {
    "safe": (
        "This prescription appears within normal dosage guidelines and no "
        "significant drug interactions were detected."
    ),
    "overdose": (
        "⚠️  The dosage specified exceeds the recommended safe threshold. "
        "This could lead to toxicity. Please verify with prescribing physician."
    ),
    "drug_interaction": (
        "⚠️  Two or more drugs in this prescription may interact dangerously. "
        "Please consult a pharmacist or physician before dispensing."
    ),
}

# Severity colours (returned for the frontend to use)
SEVERITY = {
    "safe":             "green",
    "overdose":         "red",
    "drug_interaction": "orange",
}


class PrescriptionPredictor:
    """Load fine-tuned DistilBERT model and run inference."""

    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                "Run  python train.py  first to train and save the model."
            )

        self.device    = torch.device("cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load label map from saved config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        # id2label keys are strings in JSON
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}

        self.max_len = 128

    def predict(self, text: str) -> dict:
        """
        Returns a dict:
        {
          "label":         str,
          "confidence":    float,
          "probabilities": { label: float, ... },
          "explanation":   str,
          "severity":      str,
        }
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids,
                                attention_mask=attention_mask).logits

        probs     = F.softmax(logits, dim=1).squeeze(0)
        pred_id   = probs.argmax().item()
        pred_label = self.id2label[pred_id]
        confidence = round(probs[pred_id].item(), 4)

        prob_dict = {
            self.id2label[i]: round(probs[i].item(), 4)
            for i in range(len(self.id2label))
        }

        return {
            "label":         pred_label,
            "confidence":    confidence,
            "probabilities": prob_dict,
            "explanation":   EXPLANATIONS.get(pred_label, ""),
            "severity":      SEVERITY.get(pred_label, "grey"),
        }
