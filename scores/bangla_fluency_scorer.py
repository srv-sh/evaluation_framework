import os
import json
import ast
from openai import OpenAI
from sklearn.metrics import mean_absolute_error, classification_report
import re


class banglaFluencyScorer:
    def __init__(self):
        self.client = OpenAI()

    def extract_dict_from_response(self,text):
        """
        Extracts the first dictionary-like content from model output.
        Returns None if parsing fails.
        """
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if not match:
                raise ValueError("No dictionary found in response.")
            return ast.literal_eval(match.group())
        except Exception as e:
            print(f"⚠️ Error parsing model output. Raw output: {text}")
            print(f"   ↳ Parse error: {str(e)}")
            return None

    def generate_eval_prompt(self, text):
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert Bangla language evaluator. "
                    "Given a sentence, evaluate the following dimensions:\n"
                    "1. Grammar (1 to 5)\n"
                    "2. Tone (1 to 5)\n"
                    "3. Style (1 to 5)\n"
                    "4. Register (1 to 5, formal vs informal appropriateness)\n\n"
                    "Respond with a Python dictionary like:\n"
                    "{'grammar': 5, 'tone': 4, 'style': 5, 'register': 3}"
                ),
            },
            {
                "role": "user",
                "content": f"Sentence: {text}",
            },
        ]
        
# ----------------------⚙️ EVALUATION FUNCTION --------------------------

    def evaluate_text(self,text):
        messages = self.generate_eval_prompt(text)
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content

        scores = self.extract_dict_from_response(content)

        if scores:
            for k in ["grammar", "tone", "style", "register"]:
                if k not in scores:
                    print(f"Missing score for '{k}' in: {scores}")
                    scores[k] = None
        else:
            scores = {"grammar": None, "tone": None, "style": None, "register": None}

        return scores, content

# ----------------------📂 LOAD DATASET --------------------------

    def load_dataset(self,path="/home/srv/Work/verbex/evaluation_framework/dataset/bangla_language_proficiency.json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# ----------------------📊 RUN EVALUATION --------------------------

    def evaluate_all(self,path):
        dataset = self.load_dataset(path)

        y_true = {"grammar": [], "tone": [], "style": [], "register": []}
        y_pred = {"grammar": [], "tone": [], "style": [], "register": []}
        logs = []

        for item in dataset:
            text = item["text"]
            gold = item["scores"]
            print(f"\nEvaluating: {text[:50]}...")

            scores, raw_output = self.evaluate_text(text)

            entry = {
                "text": text,
                "gold_scores": gold,
                "predicted_scores": scores,
                "raw_model_output": raw_output,
            }

            for k in y_true.keys():
                if scores[k] is not None:
                    y_true[k].append(gold[k])
                    y_pred[k].append(scores[k])
                else:
                    print(f"⚠️ No valid predictions for {k}")

            logs.append(entry)

        # Print score
        print("\n📊 Evaluation Results (Mean Absolute Error):")
        for k in y_true:
            if y_true[k]:
                mae = mean_absolute_error(y_true[k], y_pred[k])
                print(f"   {k.capitalize()}: MAE = {mae:.2f}")
            else:
                print(f"   {k.capitalize()}: No valid predictions.")

        # Save results
        with open("output/bangla_proficiency_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
