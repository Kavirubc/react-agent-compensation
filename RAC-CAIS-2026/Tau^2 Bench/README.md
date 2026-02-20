# Benchmark Data - Feb 19, 2026

This directory contains organized trace files and prompts from the τ²-bench airline domain evaluation.

## Structure

- **Task-X/**: Contains trace files for specific tasks (0, 1, 3, 5, 8, etc.).
  - `trace_{framework}_{task_id}_...json`: Detailed execution trace.

- **prompts/**: Contains the system prompts used by each framework.
  - `langgraph_prompt.txt`: Baseline ReAct agent prompt.
  - `rac_prompt.txt`: React-Agent-Compensation prompt with persistence instructions.
  - `prompt_engineer_langgraph_prompt.txt`: Enhanced prompt with error handling and compensation instructions.
  - `sagallm_planning_prompt.txt`: Prompt for SagaLLM's planning phase.
  - `sagallm_execution_prompt.txt`: Prompt for SagaLLM's execution phase.

## Frameworks
- **LangGraph**: Vanilla ReAct agent.
- **RAC**: ReAct agent with automatic rollback and retries.
- **Prompt Engineer (PE)**: LangGraph with improved error handling instructions.
- **SagaLLM**: Plan-Execute-Compensate agent.
