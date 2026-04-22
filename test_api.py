"""
test_api.py  –  Quick sanity-check against the running Flask API.
Usage:
    python test_api.py              # runs all built-in tests
    python test_api.py --text "..."  # test a single custom prescription
"""

import json
import argparse
import requests

API = "http://localhost:5000"

TESTS = [
    # (description, text, expected_label)
    (
        "Standard amoxicillin course",
        "Patient prescribed Amoxicillin 500mg three times daily for 7 days "
        "for a bacterial infection. No known allergies.",
        "safe",
    ),
    (
        "Normal ibuprofen dose",
        "Take Ibuprofen 400mg twice daily with food for mild arthritis pain.",
        "safe",
    ),
    (
        "Paracetamol massive overdose",
        "Administer Paracetamol 5000mg every 4 hours for chronic pain. "
        "Patient weighs 65kg.",
        "overdose",
    ),
    (
        "Lisinopril extreme dose",
        "Lisinopril 80mg once daily for hypertension management.",
        "overdose",
    ),
    (
        "Warfarin + Aspirin interaction",
        "Patient on Warfarin 5mg once daily. New script for Aspirin 75mg "
        "twice daily for cardiac protection.",
        "drug_interaction",
    ),
    (
        "Sertraline + Codeine interaction",
        "Continue Sertraline 50mg daily and start Codeine 30mg every 6 hours "
        "for post-operative pain.",
        "drug_interaction",
    ),
]


def check_health():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        r.raise_for_status()
        print(f"✅  Health check OK  →  {r.json()}\n")
        return True
    except Exception as e:
        print(f"❌  Cannot reach backend at {API}: {e}")
        print("   → Make sure you ran:  cd backend && python app.py\n")
        return False


def run_test(desc, text, expected):
    payload = {"text": text}
    r = requests.post(f"{API}/predict", json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()

    label = data["label"]
    conf  = data["confidence"]
    ok    = "✅" if label == expected else "⚠️ "

    print(f"{ok}  {desc}")
    print(f"   Got: {label:20s} (conf={conf:.3f})  Expected: {expected}")
    if label != expected:
        print(f"   Probabilities: {json.dumps(data['probabilities'])}")
    print()
    return label == expected


def main(args):
    print("=" * 55)
    print("  RxGuard API Tests")
    print("=" * 55 + "\n")

    if not check_health():
        return

    if args.text:
        r = requests.post(f"{API}/predict",
                          json={"text": args.text}, timeout=10)
        d = r.json()
        print(json.dumps(d, indent=2))
        return

    passed = sum(run_test(*t) for t in TESTS)
    total  = len(TESTS)
    print(f"{'='*55}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None,
                        help="Single prescription text to test")
    main(parser.parse_args())
