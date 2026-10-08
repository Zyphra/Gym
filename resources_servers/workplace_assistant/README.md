# Description

1. Environment: This is a tool use - multi step agentic environment that tests the agents ability to execute tasks in a workplace setting. Workplace assistant contains a sandbox environment with five databases, 27 tools, and 690 tasks. These tasks represent common business activities, such as sending emails and scheduling meetings.
2. Domain: Business activities
3. Source of prompts: 
- Full set of prompts (1260): https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant
4. Example prompt: Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'


Commands - 
Spin up server:

```
gym env start \
    --model-type openai_model \
    --resources-server workplace_assistant
```

Collect trajectories:
```
gym eval run --no-serve \
    --agent workplace_assistant_simple_agent \
    --input resources_servers/workplace_assistant/data/train.jsonl \
    --output results/workplace_assistant_trajectory_collection.jsonl \
   --limit 1
```

## State equivalence

The `workplace-state-equivalence-v1` grader compares all final tables as row
multisets, preserving every field and duplicate-row count. Seeded records retain
their IDs. Newly created records match by content because creation order changes
their generated IDs. Replay tracks removed seeded identities so a later reused
ID remains a new record. Duplicate IDs are invalid. These tables have no foreign
keys between generated records; adding such links requires extending the mapping.

The existing case rules remain: `status`, `list_name` and `board` are case
sensitive; other strings are compared in lowercase. This change accepts
independent forwards, creations and plots in either order while rejecting wrong
recipients, record IDs, fields, omitted writes and unintended changes. Final
answers and evidence-read order remain outside the state grader.

## Generating Additional Training Data

To generate your own training JSONL for this environment using NeMo Data Designer, see the [synthetic data generation example](notebooks/synthetic-data-generation/).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
