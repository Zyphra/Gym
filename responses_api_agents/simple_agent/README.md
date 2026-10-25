# Description

`max_steps` bounds interactive model turns. Optional
`final_answer_after_max_steps` defaults to `false`, preserving existing callers.
When enabled with a positive `max_steps`, exhaustion allows exactly one extra
model request with `tools: []` and `tool_choice: none`. For example, 24
interactive turns still permit all 24 tool turns, followed by at most one
25th answer request. An earlier answer ends normally. This does not enforce a
smaller tool-call allowance from prompt prose.

Calls emitted on the extra answer request receive ordinary error feedback and
are never dispatched. No answer or reward is fabricated. A missing final answer
remains incomplete; existing native context/output endings, unrelated HTTP
errors and cancellation retain their original handling.

Twenty-two changed deterministic CPU controls cover real loopback dispatch,
default behavior, exhausted and early answers, unexpected final calls, context
exhaustion, initiating HTTP errors, cancellation and caller-to-native scalar
verification (gold/no-op/wrong = 1/0/0). No model, judge or Python sandbox is
substituted for an outcome claim: the tool route is a deterministic fixture,
and model compliance and sandbox qualification remain separate.

Training integrations may provide `_ng_policy_completion_url` in response metadata.
The simple agent notifies it after the final policy response and before verification;
SWE and terminal harnesses notify after their agent process finishes. This is one
bounded HTTP attempt. An unavailable notification does not skip grading; the
training owner retains capacity until episode settlement. The receiver must reject
further policy turns after accepting completion and make duplicate events safe.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
