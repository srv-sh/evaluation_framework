import os
import json
import openai
from sentence_transformers import SentenceTransformer, util

class TaskExecutionAccuracy:
    def __init__(self):
        

         # Load sentence similarity model
        self.sim_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


    # Optional: Load dataset
    def load_dataset(self,path="/home/srv/Work/verbex/evaluation_framework/dataset/task_exec_dataset.json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Query OpenAI for response
    def get_llm_response(self,prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response['choices'][0]['message']['content'].strip()

    # Evaluate similarity
    def evaluate_similarity(self,expected, predicted):
        emb1 = self.sim_model.encode(expected, convert_to_tensor=True)
        emb2 = self.sim_model.encode(predicted, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()
        return round(score, 3)

    # Main evaluation
    def evaluate_all(self,path):
        dataset = self.load_dataset(path)
        results = []
        for item in dataset:
            prompt = item["prompt"]
            expected = item["expected_response"]

            print(f"📝 Evaluating prompt: {prompt[:50]}...")

            predicted = self.get_llm_response(prompt)
            similarity = self.evaluate_similarity(expected, predicted)

            result = {
                "prompt": prompt,
                "expected_response": expected,
                "llm_response": predicted,
                "similarity_score": similarity
            }

            print(f"   ➤ Similarity Score: {similarity}")
            results.append(result)

        # Save log
        with open("output/task_execution_eval_log.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Accuracy Summary
        avg_score = sum(r["similarity_score"] for r in results) / len(results)
        print(f"\n✅ Average Task Execution Score: {avg_score:.3f}")

