import openai
import re
from typing import List, Dict
import json

class conversationFluency():
    def __init__(self):
        self.prompt = """
    নিচে একটি ব্যক্তির সাথে এজেন্টের বাংলা কথোপকথন দেওয়া হয়েছে।

    তুমি একজন ভাষা বিশ্লেষক হিসেবে কাজ করছো। বিশ্লেষণ করো পুরো কথোপকথনটি কতটা প্রাকৃতিক, সাবলীল, ধারাবাহিক এবং মানবসুলভ ছিল।

    ০ থেকে ৫ এর মধ্যে একটি ফ্লুয়েন্সি স্কোর দাও এবং সংক্ষিপ্ত ব্যাখ্যা লেখো:

    স্কোরের মানে:
    ০: অপ্রাসঙ্গিক, অপ্রাকৃতিক, অস্পষ্ট কথোপকথন।
    ১: অনেক সমস্যা আছে, উত্তরের ধারাবাহিকতা নেই।
    ২: কিছুটা যুক্তিযুক্ত, কিন্তু কৃত্রিম বা এলোমেলো।
    ৩: মোটামুটি ঠিক আছে, কিছু অসংগতি আছে।
    ৪: সাবলীল ও মানবসুলভ, ছোটখাটো ত্রুটি।
    ৫: খুবই প্রাকৃতিক ও মসৃণ কথোপকথন।

    **ফলাফল অবশ্যই নিচের ফরম্যাটে JSON হিসেবে দাও:**
    ```json
    {{
      "fluency_score": <0-5>,
      "explanation": "<১-২ লাইনে কারণ>"
    }}
    ```

    কথোপকথন:
    {}
"""     
        self.model = "gpt-3.5-turbo"
        self.temperature = 0 


    def format_dialogue(self,dialogue: List[Dict[str, str]]) -> str:
        formatted = ""
        for turn in dialogue:
            formatted += f"ব্যক্তি: {turn['user']}\nএজেন্ট: {turn['agent']}\n"
        return formatted.strip()

    def score_multi_turn_conversation(self,dialogue: List[Dict[str, str]]) -> Dict[str, object]:
        dialogue_str = self.format_dialogue(dialogue)
        prompt_filled = self.prompt.format(dialogue_str)


        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user","content": prompt_filled.strip()}],
                temperature=self.temperature
            )

            content = response['choices'][0]['message']['content']
            print(content)
            
            json_str_match = re.search(r'\{[\s\S]*?\}', content)
            if json_str_match:
                parsed = json.loads(json_str_match.group())
                score = parsed.get("fluency_score", -1)
                reason = parsed.get("explanation", "No explanation provided.")
            else:
                score = -1
                reason = "Could not extract JSON from response."

            return {
                "score": score,
                "reason": reason,
                "raw_response": content
            }

        except Exception as e:
            return {
                "score": -1,
                "reason": f"Error: {str(e)}",
                "raw_response": None
            }
        
    def load_data(self,data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    
    def evaluate_all(self, data_path):
        dataset = self.load_data(data_path=data_path)
        results = []

        for item in dataset:
            dialogue = item["dialogue"]
            expected = item["expected_fluency_score"]

            result = self.score_multi_turn_conversation(dialogue)
            print("Evaluating multi-turn conversation...")
            for turn in dialogue:
                print(f"👤 {turn['user']}")
                print(f"🤖 {turn['agent']}")
            print(f"✅ Predicted Score: {result['score']}, 🎯 Expected: {expected}")
            print(f"📌 Explanation: {result['reason']}\n{'-'*60}")

            results.append({
                "dialogue": dialogue,
                "expected_score": expected,
                "predicted_score": result["score"],
                "reason": result["reason"],
                "raw_response": result["raw_response"]
            })

        with open("output/multi_turn_fluency_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


