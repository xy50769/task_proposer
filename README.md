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