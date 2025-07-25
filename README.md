# 🇧🇩 Bangla LLM Evaluation Framework

## 🧠 Project Summary

As Bangla-speaking users increasingly engage with AI systems, ensuring these systems communicate effectively, ethically, and intelligently becomes essential. This project introduces a **structured evaluation framework** to measure how well Bangla-language LLM agents perform across **seven key dimensions**—ranging from fluency and tool usage to ethical safeguards and language proficiency.

We combine real-world datasets, curated failure cases, and scoring logic powered by modern NLP techniques to validate agent performance with both depth and clarity.

---

## 📦 Evaluation Dimensions

Each evaluation module tells a unique story—revealing how the AI thinks, acts, and reacts in Bangla.

### 🗣️ 1. Conversation Fluency

> **Story**: Imagine chatting with a Bangla-speaking AI friend. You expect the conversation to feel natural—like talking to a real person. Does the AI use smooth transitions? Does it respond coherently?  
>
> This module evaluates how **fluent and human-like** those multi-turn Bangla conversations are. A specialized model assigns scores to dialogues based on how well the AI speaks the language, using metrics like flow, coherence, and linguistic smoothness.

### 🛠️ 2. Tool Calling Performance

> **Story**: Suppose you ask the AI, *“আজকের ঢাকার আবহাওয়া কেমন?”* or *“৫৩৮ ভাগ ৬ কতো?”* The smart agent should call the **right tool**—a weather fetcher or a calculator—not just reply randomly.  
>
> This module tests whether the agent **detects the correct intent** and invokes the right external tools. Using a custom callback logger, we track tool usage and compare it to the expected outcome for each prompt.

### ⚠️ 3. Guardrails Compliance

> **Story**: What if someone asks the AI for dangerous advice or uses offensive language? A responsible system must know when **not to answer** or how to reply safely.  
>
> This module checks whether the AI respects **ethical and safety guidelines**. For prompts that test boundaries (e.g., hate speech, medical advice, political bias), the agent must detect violations and respond appropriately. The system uses structured outputs to indicate whether each prompt breaches a guardrail—and compares these to labeled truths.

### ❓ 4. Edge Case Handling

> **Story**: Sometimes users ask confusing or strange questions—*“একটা বাঘ যদি গনিতে ভালো হয় তাহলে কি হবে?”* How does the AI respond when the input is ambiguous, nonsensical, or unexpected?  
>
> This module challenges the AI with **tricky Bangla prompts**, comparing its replies against safe and logical references using sentence-level similarity. It reveals how robustly the system manages uncertainty or edge cases in everyday interactions.

### 📋 5. Special Instruction Adherence

> **Story**: Sometimes, it's not *what* you say—but *how* you say it. You might want the AI to reply **without explanation**, or **in bullet points**, or even **only in Bangla numbers**. Will it follow those exact instructions?  
>
> This module verifies the agent’s ability to **follow detailed, prompt-specific instructions**. It evaluates both the semantic content (using embeddings) and structural compliance (via regex checks) to score whether the response obeys the special rules.

### 🧾 6. Task Execution Accuracy

> **Story**: You ask the AI to summarize a Bangla news article, translate a sentence, or generate a proverb. Did it complete the task **accurately**? Did the output match expectations?  
>
> This module tests how well the model performs **real-world Bangla-language tasks**, comparing its output to expected results using sentence embeddings and cosine similarity. This reflects how useful the model is for practical, task-oriented scenarios.

### 📝 7. Language Proficiency (Bangla)

> **Story**: Beyond fluency, how *good* is the AI’s Bangla? Is the grammar correct? Is the tone formal or friendly? Is the style native-sounding?  
>
> This module scores Bangla text generated by the AI across four dimensions: **grammar, tone, style, and register**. Using structured prompts to GPT-4 and comparing against expert annotations, the system calculates the **Mean Absolute Error (MAE)**—giving a clear signal of how closely the AI mirrors a native Bangla speaker.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (recommended: 3.9 or 3.10)
- **OpenAI API Key** with GPT-4 access
- **Git** for cloning the repository
- **Virtual environment** (recommended)

###  Clone the Repository

```bash
git clone https://github.com/srv-sh/evaluation_framework.git
cd evaluation_framework
```

### Build environment and activate
```
python -m venv eva_frame_env
source eva_frame_env/bin/activate
```

### Install Dependencies
Ensure you're using Python 3.8+.

```bash
pip install -r requirements.txt
```

###  Prepare the Datasets
Create a folder called dataset/ and include the following .json files: 

| Evaluation Dimension       | Dataset Path                                |
|---------------------------|---------------------------------------------|
| Conversation Fluency       | `dataset/bangla_conversation_dataset.json`  |
| Tool Calling Performance   | `dataset/tool_call_dataset.json`             |
| Guardrails Compliance      | `dataset/guardrails_dataset.json`            |
| Edge Case Handling         | `dataset/edge_case_handleing.json`            |
| Special Instruction        | `dataset/special_instruction_dataset.json`  |
| Language Proficiency       | `dataset/bangla_language_proficiency.json`  |
| Task Execution Accuracy    | `dataset/task_exec_dataset.json`             |

### Exporting OpenAI API key
```
export OPENAI_API_KEY="your_api_key_here"
```

### Run main script
```

mkdir output
python main.py

```


### Output Directory
After executing code you will get directory like below
```bash
output/
├── conversation_fluency.json
├── tool_calling_performance.json
├── guardrails.json
├── edge_case_handling.json
├── special_instruction.json
├── bangla_language_proficiency.json
├── task_execution_accuracy.json

```

