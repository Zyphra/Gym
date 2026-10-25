# Description


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
