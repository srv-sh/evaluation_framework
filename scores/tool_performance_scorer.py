import json
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from langchain.tools import Tool
from tools.tools import calculator_tool, weather_tool, fluency_tool
from langchain.callbacks.base import BaseCallbackHandler


# Custom callback to log which tools are called
class ToolCallLogger(BaseCallbackHandler):
    def __init__(self):
        self.tool_calls = []

    def on_tool_start(self, tool, input_str, **kwargs):
        # Handle both Tool object and dict case
        tool_name = getattr(tool, "name", None)
        if tool_name is None and isinstance(tool, dict):
            tool_name = tool.get("name", "Unknown")
        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str
        })

    def get_tool_calls(self):
        return self.tool_calls

    def reset(self):
        self.tool_calls = []



class toolCallingPerformance:
    def __init__(self):
        # LLM and Tracer setup
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # Tools setup
        self.tools = [
            Tool(
                name="Calculator",
                func=calculator_tool,
                description="Evaluates a simple math expression and returns the result."
            ),
            Tool(
                name="Weather",
                func=weather_tool,
                description="Returns weather info for a given location."
            ),
            Tool(
                name="Fluency",
                func=fluency_tool,
                description="Scores fluency of a Bangla sentence."
            ),
        ]

        # Agent setup
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
        )


    # Load dataset from JSON
    def load_dataset(self,path="/home/srv/Work/verbex/evaluation_framework/dataset/tool_call_dataset.json"):
        with open(path, "r") as f:
            return json.load(f)


    # Evaluate a single prompt with tool tracking
    def evaluate_prompt(self,prompt_data):
        prompt = prompt_data["prompt"]
        expected_tool = prompt_data["expected_tool"]
        expected_substring = prompt_data["expected_substring"]

        print(f"Evaluating prompt: {prompt}")

        tool_logger = ToolCallLogger()

        try:
            result = self.agent_executor.run(prompt, callbacks=[tool_logger])
            # passed = expected_substring in result
            tools_used = tool_logger.get_tool_calls()
        except Exception as e:
            result = f"[ERROR] {str(e)}"
            passed = False
            tools_used = []

        return {
            "prompt": prompt,
            "expected_tool": expected_tool,
            "expected_substring": expected_substring,
            "agent_response": result,
            "tools_used": tools_used,
        }


    # Evaluate all prompts in the dataset
    def evaluate_all(self,data_path):
        dataset = self.load_dataset(path=data_path)
        results = [self.evaluate_prompt(p) for p in dataset]
        total = len(results)
        

        success_count=0
        # Log mismatches
        for r in results:
            used_tool_names = [tool["tool"] for tool in r["tools_used"]]
            if r["expected_tool"] in used_tool_names:
                success_count += 1
        print(f"\n✅ Tool Call Accuracy: {success_count}/{total} ({(success_count/total)*100:.2f}%)\n")


        # Optional: save logs
        with open("output/tool_call_eval_log.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
