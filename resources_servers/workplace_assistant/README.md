# Description

1. Environment: This is a tool use - multi step agentic environment that tests the agents ability to execute tasks in a workplace setting. Workplace assistant contains a sandbox environment with five databases and 27 tools. Its tasks represent common business activities, such as sending emails and scheduling meetings. Task counts depend on the source population; see Task populations below.
2. Domain: Business activities
3. Source of prompts: 
- Prompts: https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant
4. Example prompt: Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'


## Task populations

The upstream description's figures of 690 tasks and 1,260 prompts are not row
counts of the pinned splits. The committed dataset metrics
(`data/train_metrics.json`, `data/validation_metrics.json`) record 1,255 TRAIN and
545 validation examples. A training source built from them has its own counts:
the RLTasks
[`workbench-full-stock-plus-hard-v1`](https://github.com/Zyphra/rl_tasks/blob/wrapper/datasets/workbench-full-stock-plus-hard-v1.yaml)
recipe starts from the 1,255 TRAIN rows, removes 140 held-out parents and 177
full-reward no-op rows, and adds 235 generated rows, giving 938 stock plus 235
generated (1,173) rows. The
[RLTasks WorkBench binding README](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/nemo_gym/training/workbench/README.md)
tabulates each population.

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

Sent email is graded as sent, not as the final mailbox shows it. Replay records
every email each action creates (send, reply, forward), and the predicted and
reference replays must send the same multiset of emails by content. Deleting a
wrong email does not unsend it, so a wrong send followed by a delete and a
correct resend fails even though the final mailbox matches.

Calendar start times and durations compare their validated `Timestamp` and
integer-minute values, so equivalent ISO timestamps and leading-zero durations
are accepted. Naive and timezone-aware timestamps remain distinct; different
times or durations still fail.

## Generating Additional Training Data

To generate your own training JSONL for this environment using NeMo Data Designer, see the [synthetic data generation example](notebooks/synthetic-data-generation/).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
