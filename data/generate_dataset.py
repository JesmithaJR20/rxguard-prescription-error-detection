"""
generate_dataset.py
Generates a synthetic medical prescription dataset for training.
Labels: safe | overdose | drug_interaction
"""

import csv
import random

# ─────────────────────────────────────────────
#  Template pools
# ─────────────────────────────────────────────

SAFE_TEMPLATES = [
    "Patient prescribed {drug} {safe_dose} {freq} for {condition}.",
    "Administer {drug} {safe_dose} orally {freq}. Indication: {condition}.",
    "{drug} {safe_dose} {freq} as needed for {condition}. Max {safe_dose} per dose.",
    "Start {drug} {safe_dose} {freq} for {condition}. Monitor after 2 weeks.",
    "Take {drug} {safe_dose} with food {freq} for {condition}.",
    "{drug} {safe_dose} {freq} prescribed. No known allergies reported.",
    "Prescription: {drug} {safe_dose} {freq} for mild {condition}.",
    "{drug} {safe_dose} once daily at bedtime for {condition}.",
    "Patient to take {drug} {safe_dose} {freq} with plenty of water.",
    "{drug} {safe_dose} {freq} for 7 days. Refill x1 for {condition}.",
]

OVERDOSE_TEMPLATES = [
    "Patient prescribed {drug} {high_dose} {freq} for {condition}.",
    "Administer {drug} {high_dose} every 4 hours for {condition}.",
    "{drug} {high_dose} three times daily. Exceeds maximum safe dosage.",
    "Take {drug} {high_dose} {freq} — dose is critically high for {condition}.",
    "Prescribed {drug} {high_dose} orally. Standard max is {safe_dose}.",
    "{drug} {high_dose} {freq} ordered. Potential overdose risk detected.",
    "Administer {drug} {high_dose} as loading dose without tapering.",
    "{drug} {high_dose} daily for chronic {condition} — far above guidelines.",
    "Prescription calls for {drug} {high_dose} every 2 hours — unsafe.",
    "{drug} {high_dose} every 6 hours; maximum recommended is {safe_dose} per day.",
]

INTERACTION_TEMPLATES = [
    "Patient taking {drug1} {safe_dose} and {drug2} {safe_dose2} simultaneously.",
    "Co-prescribe {drug1} with {drug2} for {condition}.",
    "{drug1} {safe_dose} {freq} along with {drug2} {safe_dose2} daily.",
    "Add {drug2} {safe_dose2} to existing regimen of {drug1} {safe_dose}.",
    "Patient on {drug1}; new prescription for {drug2} for {condition}.",
    "{drug1} and {drug2} prescribed together — potential serious interaction.",
    "Continue {drug1} {safe_dose} and start {drug2} {safe_dose2} concurrently.",
    "Combination therapy: {drug1} {safe_dose} + {drug2} {safe_dose2} for {condition}.",
    "Patient on long-term {drug1}; prescribe {drug2} for acute {condition}.",
    "{drug1} {safe_dose} and {drug2} {safe_dose2} co-administered. Check interaction.",
]

# ─────────────────────────────────────────────
#  Drug / Dose data
# ─────────────────────────────────────────────

DRUGS = [
    {"name": "Ibuprofen",    "safe": "400mg",   "high": "2400mg"},
    {"name": "Amoxicillin",  "safe": "500mg",   "high": "3000mg"},
    {"name": "Metformin",    "safe": "500mg",   "high": "3500mg"},
    {"name": "Lisinopril",   "safe": "10mg",    "high": "80mg"},
    {"name": "Atorvastatin", "safe": "20mg",    "high": "120mg"},
    {"name": "Paracetamol",  "safe": "500mg",   "high": "5000mg"},
    {"name": "Omeprazole",   "safe": "20mg",    "high": "120mg"},
    {"name": "Sertraline",   "safe": "50mg",    "high": "400mg"},
    {"name": "Aspirin",      "safe": "75mg",    "high": "1500mg"},
    {"name": "Codeine",      "safe": "30mg",    "high": "300mg"},
    {"name": "Warfarin",     "safe": "5mg",     "high": "30mg"},
    {"name": "Digoxin",      "safe": "0.125mg", "high": "1.5mg"},
    {"name": "Levothyroxine","safe": "50mcg",   "high": "600mcg"},
    {"name": "Amlodipine",   "safe": "5mg",     "high": "50mg"},
    {"name": "Gabapentin",   "safe": "300mg",   "high": "4800mg"},
]

# Pairs that have known dangerous interactions
INTERACTION_PAIRS = [
    ("Warfarin",     "Aspirin",       "75mg",    "75mg"),
    ("Sertraline",   "Tramadol",      "50mg",    "50mg"),
    ("Lisinopril",   "Spironolactone","10mg",    "25mg"),
    ("Metformin",    "Alcohol",       "500mg",   "unknown"),
    ("Amoxicillin",  "Warfarin",      "500mg",   "5mg"),
    ("Codeine",      "Sertraline",    "30mg",    "50mg"),
    ("Ibuprofen",    "Warfarin",      "400mg",   "5mg"),
    ("Digoxin",      "Amiodarone",    "0.125mg", "200mg"),
    ("Gabapentin",   "Codeine",       "300mg",   "30mg"),
    ("Aspirin",      "Ibuprofen",     "75mg",    "400mg"),
    ("Levothyroxine","Calcium",       "50mcg",   "500mg"),
    ("Atorvastatin", "Clarithromycin","20mg",    "250mg"),
    ("Amlodipine",   "Simvastatin",   "5mg",     "40mg"),
    ("Warfarin",     "Omeprazole",    "5mg",     "20mg"),
    ("Metformin",    "Contrast Dye",  "500mg",   "IV dose"),
]

CONDITIONS = [
    "hypertension", "type 2 diabetes", "bacterial infection",
    "chronic pain", "anxiety disorder", "acid reflux",
    "high cholesterol", "atrial fibrillation", "hypothyroidism",
    "depression", "epilepsy", "osteoarthritis",
    "heart failure", "migraine", "urinary tract infection",
]

FREQUENCIES = ["twice daily", "once daily", "three times daily",
               "every 8 hours", "every 12 hours", "as needed"]


def make_safe(n):
    rows = []
    for _ in range(n):
        drug = random.choice(DRUGS)
        tmpl = random.choice(SAFE_TEMPLATES)
        cond = random.choice(CONDITIONS)
        freq = random.choice(FREQUENCIES)
        text = tmpl.format(
            drug=drug["name"], safe_dose=drug["safe"],
            freq=freq, condition=cond
        )
        rows.append({"text": text, "label": "safe"})
    return rows


def make_overdose(n):
    rows = []
    for _ in range(n):
        drug = random.choice(DRUGS)
        tmpl = random.choice(OVERDOSE_TEMPLATES)
        cond = random.choice(CONDITIONS)
        freq = random.choice(FREQUENCIES)
        text = tmpl.format(
            drug=drug["name"], high_dose=drug["high"],
            safe_dose=drug["safe"], freq=freq, condition=cond
        )
        rows.append({"text": text, "label": "overdose"})
    return rows


def make_interaction(n):
    rows = []
    for _ in range(n):
        pair = random.choice(INTERACTION_PAIRS)
        drug1, drug2, dose1, dose2 = pair
        tmpl = random.choice(INTERACTION_TEMPLATES)
        cond = random.choice(CONDITIONS)
        freq = random.choice(FREQUENCIES)
        text = tmpl.format(
            drug1=drug1, drug2=drug2,
            safe_dose=dose1, safe_dose2=dose2,
            freq=freq, condition=cond
        )
        rows.append({"text": text, "label": "drug_interaction"})
    return rows


def main():
    random.seed(42)
    per_class = 160  # ~480 total, balanced

    data = make_safe(per_class) + make_overdose(per_class) + make_interaction(per_class)
    random.shuffle(data)

    out_path = "data/prescriptions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✅  Dataset saved → {out_path}  ({len(data)} rows)")

    # Quick label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in data)
    for lbl, cnt in sorted(dist.items()):
        print(f"   {lbl:20s}: {cnt}")


if __name__ == "__main__":
    main()
