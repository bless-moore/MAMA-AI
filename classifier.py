"""Pathway 1: warning-sign classifier.

Two layers, so a missed danger sign is unlikely:
  1. Deterministic regex rules for well-known danger signs (high recall, English only).
  2. TF-IDF + logistic regression over the 4 labels, with a low emergency threshold
     (we prefer false alarms to missed emergencies).

Usage:
    python classifier.py --data data/dataset.jsonl
"""
import argparse
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from common import LABELS, load_jsonl, split

ARTIFACT = Path("artifacts/classifier.joblib")
EMERGENCY_THRESHOLD = 0.30

EMERGENCY_RULES = [
    # headache / vision (pre-eclampsia)
    r"\b(severe|bad|terrible|worst|splitting|pounding)\b.{0,30}\bheadache|headache\b.{0,30}\b(will not|won'?t|does not|doesn'?t) (go|stop|pass)",
    r"blurr?(ed|y)\s+(vision|sight)|(vision|sight|eyes?)\b.{0,15}\bblurr?(ed|y)|can'?t see (well|properly|clearly)|flashing lights",
    # fits, collapse
    r"\b(had|having|has|got|get|getting|just had) (a )?fits?\b|\bfits\b|\b(convulsion|convulsions|convulsing|seizure|seizures)\b",
    r"\bunconscious\b|\bfainted\b|\bpassed out\b|\bpassing out\b|\bcollaps(e|ed|ing)\b|(not|isn'?t) responding|unresponsive",
    # bleeding
    r"heavy\b.{0,25}\bbleeding|bleeding\b.{0,30}\b(heavy|heavily|a lot|soak)|soaking\b.{0,25}\b(pad|pads|clothes)|a lot of blood|bleeding\b.{0,25}(will not|won'?t|does not|doesn'?t) stop",
    # baby movement
    r"(baby|fetus).{0,40}(not moving|stopped moving|moving (much |far )?less|no movement|not kicking|not responding)",
    r"(not|haven'?t|have not) felt\b.{0,20}\b(baby|kick|kicks|move|moving|movement)",
    # waters / early labour
    r"(water|waters|fluid)\b.{0,20}\b(broke|broken|break|leak|leaking|gush)|gush of fluid",
    r"(contractions?|labou?r pains?|tightening)\b.{0,60}\b(\d{2} weeks|months)|labou?r\b.{0,30}\bearly",
    # breathing, chest, newborn
    r"(difficulty|trouble|struggling) (to )?(breathing|breathe|catch my breath)|can'?t breathe|chest pain|gasping|blue lips|turning blue",
    r"(newborn|baby).{0,40}(floppy|not breathing|had a fit|will not feed|blue)",
    # severe abdominal pain
    r"(severe|sudden)\b.{0,20}\b(belly|stomach|abdominal|tummy)\b.{0,15}\bpain|(belly|tummy|abdomen)\b.{0,30}\b(very hard|rock hard)",
]
_RULES = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_RULES]


def build_pipeline():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=5),
    )


def classify(pipe, text):
    """Return {'label', 'confidence', 'source'}."""
    if any(r.search(text) for r in _RULES):
        return {"label": "emergency", "confidence": 1.0, "source": "rules"}
    probs = dict(zip(pipe.classes_, pipe.predict_proba([text])[0]))
    if probs.get("emergency", 0.0) >= EMERGENCY_THRESHOLD:
        return {"label": "emergency", "confidence": float(probs["emergency"]), "source": "model"}
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": float(probs[label]), "source": "model"}


def load(path=ARTIFACT):
    return joblib.load(path)


def train(data_path):
    train_rows, val_rows = split(load_jsonl(data_path))
    pipe = build_pipeline()
    pipe.fit([r["question"] for r in train_rows], [r["label"] for r in train_rows])
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACT)

    y_true = [r["label"] for r in val_rows]
    y_pred = [classify(pipe, r["question"])["label"] for r in val_rows]
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    print(f"Saved {ARTIFACT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed_examples.jsonl")
    train(ap.parse_args().data)
