import json
import re
import os
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.tracers import LangChainTracer
class SpecialInstructionAdherence:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.80
        self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo"
            )

        self.similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.LOG_PATH = "output/instruction_eval_log.json"
# ========== Utility Functions ==========

    def load_dataset(self,path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def similarity_score(self,pred: str, ref: str) -> float:
        emb1 = self.similarity_model.encode(pred, convert_to_tensor=True)
        emb2 = self.similarity_model.encode(ref, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def regex_match(self,actual: str, pattern: str) -> bool:
        return bool(re.fullmatch(pattern.strip(), actual.strip()))

# ========== Evaluation Core ==========

    def evaluate_sample(self,sample: Dict) -> Dict:
        prompt = sample["prompt"]
        expected = sample["expected_response"]
        strict_pattern = sample.get("expected_pattern")  # Optional regex pattern

        print(f"\n🔍 Prompt: {prompt}")
        try:
            actual = self.llm.predict(prompt)
        except Exception as e:
            return {
                "prompt": prompt,
                "expected": expected,
                "actual": str(e),
                "similarity_score": 0,
                "passed": False,
                "error": True
            }

        sim_score = self.similarity_score(actual, expected)
        pattern_pass = self.regex_match(actual, strict_pattern) if strict_pattern else None

        if strict_pattern:
            passed = pattern_pass
        else:
            passed = sim_score >= self.SIMILARITY_THRESHOLD

        return {
            "prompt": prompt,
            "expected": expected,
            "actual": actual,
            "similarity_score": sim_score,
            "regex_match": pattern_pass,
            "passed": passed,
            "error": False
        }

    def evaluate_all(self,path):
        dataset = self.load_dataset(path)
        results = [self.evaluate_sample(p) for p in dataset]
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed

        print(f"\n✅ Special Instruction Adherence Accuracy: {passed}/{total} ({(passed / total) * 100:.2f}%)")

        with open(self.LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

