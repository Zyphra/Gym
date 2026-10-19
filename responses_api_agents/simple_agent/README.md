# Description

`max_steps` bounds model turns. Optional `max_tool_calls` bounds attempted
function calls per episode (including malformed or unpresented calls); its
default `null` preserves existing behavior. Once the budget is spent, subsequent
model requests use `tools: []` and `tool_choice: none`. With this option enabled,
the final allowed model turn also disables tools so it can contain an answer.
Calls emitted despite that restriction receive ordinary error feedback and are
never dispatched. Parallel calls cannot exceed the budget. No answer or reward
is fabricated; a model that still supplies no answer remains subject to the
resource verifier and the existing turn cap.

Deterministic loopback controls exercise actual agent dispatch, cookies,
trajectories and last-call-to-answer requests with no model or judge service.
They do not establish model compliance or a reward-rate improvement.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
