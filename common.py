"""Shared constants and helpers for MAMA-AI."""
import json
import random
from pathlib import Path

MODEL_ID = "google/gemma-3-1b-it"  # gated: accept the licence on Hugging Face first
LABELS = ["emergency", "urgent", "info", "out_of_scope"]

SYSTEM_PROMPT = (
    "You are MAMA-AI, a maternal health education assistant for pregnant women in Ghana. "
    "Use simple, kind, clear English. You give education only: you never diagnose, never "
    "prescribe or give medicine doses, and never predict pregnancy outcomes. If the user "
    "describes a danger sign, tell them to go to the nearest health facility immediately. "
    "Always encourage regular antenatal care with a midwife or doctor."
)

DISCLAIMER = (
    "\n\nThis is health education, not a diagnosis or a replacement for a midwife or doctor."
)


def load_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def split(rows, val_frac=0.15, seed=42):
    """Deterministic split shared by classifier, LoRA training and evaluation.

    If rows carry a "topic" key, whole topics are held out (stratified by label) so
    near-duplicate questions cannot leak from train into validation.
    """
    rows = list(rows)
    rng = random.Random(seed)
    if rows and all("topic" in r for r in rows):
        by_label = {}
        for r in rows:
            by_label.setdefault(r["label"], {}).setdefault(r["topic"], []).append(r)
        train, val = [], []
        for label in sorted(by_label):
            topics = sorted(by_label[label])
            rng.shuffle(topics)
            n_val = max(1, round(len(topics) * val_frac)) if len(topics) >= 3 else 0
            for i, t in enumerate(topics):
                (val if i < n_val else train).extend(by_label[label][t])
        return train, val
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]
