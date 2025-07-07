from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain.chat_models import ChatOpenAI

class edgeCaseHandling:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.model = SentenceTransformer("sentence-transformers/LaBSE")

    def load_dataset(self,path="/home/srv/Work/verbex/evaluation_framework/dataset/edge_case_handleing.json"):
        with open(path, "r") as f:
            return json.load(f)

    def get_similarity(self,resp1, resp2):
        emb1 = self.model.encode([resp1])[0]
        emb2 = self.model.encode([resp2])[0]
        return cosine_similarity([emb1], [emb2])[0][0] 

    def evaluate_prompt(self,prompt_data):
        prompt = prompt_data["prompt"]
        ideal = prompt_data["ideal_response"]

        print(f"\n🧪 Prompt: {prompt}")
        llm_response = self.llm.predict(prompt)
        score = self.get_similarity(llm_response, ideal)

        return {
            "prompt": prompt,
            "ideal_response": ideal,
            "llm_response": llm_response,
            "similarity_score": str(round(score, 4)),
            "passed": bool(score > 0.7)  # adjustable threshold
        }

    def evaluate_all(self,path):
        dataset = self.load_dataset(path)
        results = [self.evaluate_prompt(p) for p in dataset]
        print(results)
        passed = sum(r["passed"] for r in results)
        total = len(results)
        print(f"\n✅ Similarity Accuracy: {passed}/{total} ({(passed/total)*100:.2f}%)")

        with open("output/edge_case_similarity_log.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# if __name__ == "__main__":
#     evaluate_all()
