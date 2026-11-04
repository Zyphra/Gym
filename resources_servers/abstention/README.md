# Description

This is a resources server that trains a policy model to abstain from answering when unsure rather than hallucinating. It uses a three-tier reward scheme:

- **Correct** (1.0): The model provides a correct answer
- **Abstain** (configurable, default 0.5): The model outputs `\boxed{[IDK]}` or the LLM judge grades the answer as NOT_ATTEMPTED
- **Incorrect** (0.0): The model provides an incorrect answer

Correctness is verified by an LLM judge using the OMNISCIENCE_GRADER template instead of string matching. The judge grades the model's extracted answer against the gold target as one of CORRECT, INCORRECT, or NOT_ATTEMPTED. NOT_ATTEMPTED is reserved for an answer that proposes no concrete answer (a refusal, "I don't know", or a request for more context). A hedged answer that names a concrete answer ("I'm not sure, but it might be X") is an attempt and is graded CORRECT or INCORRECT against the gold target, so a hedge never earns the abstention reward.

The dataset used is [HotPotQA](https://hotpotqa.github.io/) (fullwiki split).

# Example usage

## Running servers

```bash
gym env start \
    --model-type openai_model \
    --resources-server abstention \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent abstention_simple_agent \
    --input resources_servers/abstention/data/example.jsonl \
    --output results/abstention_verify_responses.jsonl \
    --limit 3
```

## Preprocessing HotPotQA data

```bash
python resources_servers/abstention/dataset_preprocess.py \
    --download \
    --raw-data-dir /path/to/data/hotpotqa \
    --output-dir resources_servers/abstention/data
```

# Licensing information

Code: Apache 2.0
Data:
- HotPotQA: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies:
- nemo_gym: Apache 2.0
