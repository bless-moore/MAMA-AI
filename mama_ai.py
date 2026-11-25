"""Combined MAMA-AI pipeline (classification + conversational assistant).

Flow:
  user text -> classifier
     emergency    -> fixed, pre-approved safety message (the LLM is NOT used)
     out_of_scope -> LLM answer (trained to decline), disclaimer added
     urgent       -> LLM answer + "see a health professional today" banner
     info         -> LLM answer + disclaimer

Usage:
    python mama_ai.py "What foods should I eat in pregnancy?"
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import classifier
from common import DISCLAIMER, MODEL_ID, SYSTEM_PROMPT

# Have a clinician confirm these numbers and wording before any deployment.
EMERGENCY_REPLY = (
    "What you describe can be a danger sign in pregnancy or after birth. "
    "Please go to the nearest hospital or health facility IMMEDIATELY, or call an ambulance "
    "(112 / 193). Ask someone to go with you. Do not wait for an AI reply or for your next "
    "antenatal appointment."
)
URGENT_BANNER = (
    "\n\nPlease see a midwife or doctor TODAY about this. If it gets worse, go to the nearest "
    "health facility immediately."
)


class MamaAI:
    def __init__(self, adapter_dir="artifacts/mama-lora/adapter",
                 classifier_path=classifier.ARTIFACT, device=None):
        self.clf = classifier.load(classifier_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype, attn_implementation="eager"
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, question, max_new_tokens=300):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1
        )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def respond(self, question):
        route = classifier.classify(self.clf, question)
        label = route["label"]
        if label == "emergency":
            answer = EMERGENCY_REPLY
        elif label == "urgent":
            answer = self._generate(question) + URGENT_BANNER
        else:
            answer = self._generate(question) + DISCLAIMER
        return {"label": label, "route_source": route["source"],
                "confidence": route["confidence"], "answer": answer}


if __name__ == "__main__":
    bot = MamaAI()
    print(bot.respond(" ".join(sys.argv[1:]) or "What foods should I eat in pregnancy?"))
