import os
from scores.fluency_scorer import conversationFluency
from scores.tool_performance_scorer import toolCallingPerformance
from scores.guarfrails_scorer import guardrailsCompliance
from scores.edge_case_scorer import edgeCaseHandling
from scores.special_instruction_scorer import SpecialInstructionAdherence
from scores.bangla_fluency_scorer import banglaFluencyScorer
from scores.task_execution_scorer import TaskExecutionAccuracy

# Define dataset paths
conversation_fluency_dataset = 'dataset/bangla_conversation_dataset.json'
tool_calling_performance_dataset = 'dataset/tool_call_dataset.json'
guardrails_dataset = 'dataset/guardrails_dataset.json'
edge_case_dataset = 'dataset/edge_case_handleing.json'
special_instruction_dataset = 'dataset/special_instruction_dataset.json'
bangla_fluency_dataset = 'dataset/bangla_language_proficiency.json'
task_execution_dataset = 'dataset/task_exec_dataset.json'

# Dictionary mapping evaluation name to scorer class and dataset
evaluation_dict = {
    "conversation_fluency": {"model": conversationFluency(), "dataset": conversation_fluency_dataset},
    "tool_calling_performance": {"model": toolCallingPerformance(), "dataset": tool_calling_performance_dataset},
    "guardrails": {"model": guardrailsCompliance(), "dataset": guardrails_dataset},
    "edge_case": {"model": edgeCaseHandling(), "dataset": edge_case_dataset},
    "special_instruction": {"model": SpecialInstructionAdherence(), "dataset": special_instruction_dataset},
    "bangla_fluency": {"model": banglaFluencyScorer(), "dataset": bangla_fluency_dataset},
    "task_execution": {"model": TaskExecutionAccuracy(), "dataset": task_execution_dataset},
}

# Run evaluations
for name, evaluator in evaluation_dict.items():
    print(f"Running evaluation for: {name}")
    model = evaluator["model"]
    dataset = evaluator["dataset"]
    model.evaluate_all(dataset)
    print(f"Completed evaluation for: {name}\n")
