# webagent
Generates tasks with low medium and high difficulty levels

Supports adaptive iterations using failed task trajectories

Uses GPT or GPT Vision for analyzing failures and improving tasks

Handles screenshots as base64 images when available

Supports both text and image-based prompt workflows

# How to Use

Edit config.json to set model parameters and paths

Add examples in .jsonl format

Run python your_script.py

You can call

generate_task(t=0) for initial task generation

adaptive_task_iteration(ratio=0.5) for refining tasks using failure analysis

generate_tasks_by_domain_config(t=2) to generate tasks by domain settings

File Structure
config.json for model and generation settings

WebVoyager_test.jsonl for example tasks

failure_trajectory.json for logging failed cases

Prompt template file with three sections

# Others

Confidence Score

The confidence_score is a float between 0 and 1 used to assess how likely a task is open-ended

It is returned by a GPT classifier based on Prompt 3

1.0 means the task is clearly open-ended

0.0 means the task is clearly non-open-ended

0.5 means the model is uncertain or the task is ambiguous

Threshold

The value of confidence_threshold is set in config.json

Tasks with confidence_score greater than or equal to the threshold are skipped during task refinement

Only tasks with low confidence (below the threshold) are retained and revised